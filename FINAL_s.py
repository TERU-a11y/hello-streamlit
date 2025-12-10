# app.py
# ベンチプレス100kgチャレンジ ＋ AIメニュー管理アプリ
# Streamlit + OpenAI (ChatGPT) / Supabase 連携なし

import os
import json
import base64
from datetime import date, timedelta
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import altair as alt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================
# 定数
# =========================

TARGET_1RM = 100  # 目標ベンチプレス 100kg
WEEKDAYS_JP = ["月", "火", "水", "木", "金", "土", "日"]


# =========================
# セッションステート初期化
# =========================

def init_state():
    # プロフィール / 100kg到達予測
    if "profile" not in st.session_state:
        st.session_state.profile: Dict[str, Any] = {}

    # タンパク質関連
    if "protein_goal" not in st.session_state:
        st.session_state.protein_goal: float = 0.0
    if "protein_today" not in st.session_state:
        st.session_state.protein_today: float = 0.0
    if "protein_celebrated_date" not in st.session_state:
        st.session_state.protein_celebrated_date: Optional[str] = None

    # 100kgグラフ用ログ
    if "training_logs" not in st.session_state:
        st.session_state.training_logs: List[Dict[str, Any]] = []
    if "celebrated_100kg" not in st.session_state:
        st.session_state.celebrated_100kg: bool = False

    # ===== ここから「今週のトレーニング（元②トレーニング管理）」系 =====
    if "initial_info" not in st.session_state:
        st.session_state.initial_info = None  # AIメニュー用の初期情報
    if "weekly_plan" not in st.session_state:
        st.session_state.weekly_plan: List[Any] = []  # 各週の AI メニュー
    if "records" not in st.session_state:
        st.session_state.records: Dict[int, Dict[str, Any]] = {}  # 実績
    if "current_week" not in st.session_state:
        st.session_state.current_week: int = 1
    if "training_started" not in st.session_state:
        st.session_state.training_started: bool = False
    if "expanded_status" not in st.session_state:
        st.session_state.expanded_status: Dict[str, bool] = {}
    if "max_test_result" not in st.session_state:
        st.session_state.max_test_result: float = 0.0
    if "goal_achieved_pending" not in st.session_state:
        st.session_state.goal_achieved_pending: bool = False
    if "max_registered_not_achieved" not in st.session_state:
        st.session_state.max_registered_not_achieved: bool = False
    if "day_review_done" not in st.session_state:
        st.session_state.day_review_done: Dict[int, Dict[str, bool]] = {}
    if "last_review" not in st.session_state:
        st.session_state.last_review: Dict[int, Dict[str, str]] = {}
    if "next_week_config_pending" not in st.session_state:
        st.session_state.next_week_config_pending: bool = False


# =========================
# OpenAI クライアント
# =========================

def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None

    api_key = None
    # 1. secrets.toml
    try:
        if "openai" in st.secrets:
            api_key = st.secrets["openai"].get("api_key")
    except Exception:
        # secrets.tomlが存在しない、またはアクセスできない場合は無視
        pass
    # 2. 環境変数
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


# =========================
# 共通ロジック（100kg達成日など）
# =========================

def estimate_weeks_to_target(current_1rm: float, sessions_per_week: int) -> int:
    """100kgに到達するまでの週数をざっくり推定（簡易モデル）"""
    if current_1rm >= TARGET_1RM:
        return 0

    base_gain = 0.6
    freq_bonus = (sessions_per_week - 3) * 0.15
    weekly_gain = max(0.3, base_gain + freq_bonus)  # 最低でも0.3kg/週
    need_kg = TARGET_1RM - current_1rm
    weeks = int((need_kg / weekly_gain) + 0.999)
    return max(1, weeks)


def log_training_snapshot(note: str = "", log_date: Optional[date] = None):
    """現在の1RM等を training_logs に記録"""
    if not st.session_state.profile:
        return

    if log_date is None:
        d = date.today().isoformat()
    else:
        d = log_date.isoformat()

    log = {
        "date": d,
        "current_1rm": float(st.session_state.profile.get("current_1rm", 0.0)),
        "note": note,
    }
    st.session_state.training_logs.append(log)


def get_first_100kg_date() -> Optional[str]:
    logs = st.session_state.training_logs
    if not logs:
        return None
    sorted_logs = sorted(logs, key=lambda x: x["date"])
    for log in sorted_logs:
        if float(log.get("current_1rm", 0.0)) >= TARGET_1RM:
            return log["date"]
    return None


# =========================
# 「今週のトレーニング」用 ロジック
# =========================

def convert_records_to_dataframe(records: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """records をグラフ用の DataFrame に変換"""
    data = []
    for week_num, week_data in records.items():
        for day_of_week, day_data in week_data.items():
            date_id = f"W{week_num}_{day_of_week}"
            for exercise, record in day_data.items():
                total_load = record["weight"] * record["reps"] * record["sets"]
                data.append(
                    {
                        "Week_Day_ID": date_id,
                        "Week": week_num,
                        "Exercise": exercise,
                        "Weight": record["weight"],
                        "Reps": record["reps"],
                        "Sets": record["sets"],
                        "Total_Load": total_load,
                        "Is_Max": record.get("is_max", False),
                    }
                )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    unique_ids = df["Week_Day_ID"].unique().tolist()
    df["Week_Day_ID"] = pd.Categorical(df["Week_Day_ID"], categories=unique_ids, ordered=True)
    return df


def format_records_for_prompt(
    records: Dict[int, Dict[str, Any]],
    current_week: int,
    days: List[str],
    for_review: bool = False,
    target_day: Optional[str] = None,
) -> str:
    """前週まで / 当日実績をプロンプト用に整形"""
    prompt_records: List[str] = []

    if for_review:
        week = current_week
        if target_day in records.get(week, {}):
            day_data = f"--- {week}週目 {target_day}の当日実績 ---\n"
            for exercise, data in records[week][target_day].items():
                total_load = data["weight"] * data["reps"] * data["sets"]
                day_data += (
                    f"- {exercise}: {data['weight']}kg x {data['reps']}回 x "
                    f"{data['sets']}セット. 総負荷量: {total_load}kg\n"
                )
            prompt_records.append(day_data)

        if prompt_records:
            return "\n".join(prompt_records) + "\n\n**この日のトレーニングは完了しました。**"
        return "実績データが見つかりません。"

    # 次週用（前週の実績）
    last_week = current_week - 1
    if last_week in records and last_week >= 1:
        for day in days:
            if day in records[last_week]:
                day_data = f"--- {last_week}週目 {day}の実績 ---\n"
                for exercise, data in records[last_week][day].items():
                    day_data += (
                        f"- {exercise}: {data['weight']}kg x {data['reps']}回 x "
                        f"{data['sets']}セット. メモ: {data['note']}\n"
                    )
                prompt_records.append(day_data)
        if prompt_records:
            return "\n".join(prompt_records)

    return "前週の実績はありません。"


def generate_ai_week_plan(
    week_num: int,
    weekdays: List[str],
    initial_info: Dict[str, Any],
    records: Dict[int, Dict[str, Any]],
    max_bp: float,
    client: Any,
) -> Optional[List[Dict[str, Any]]]:
    """OpenAI APIを使用して次週のトレーニングメニューを生成"""

    if client is None:
        st.error("OpenAI APIキーが設定されていません。AIメニュー生成は利用できません。")
        return None

    freq_next = len(weekdays)
    last_week_records = format_records_for_prompt(
        records, week_num, initial_info["weekdays"]
    )

    json_schema_str = """
{
    "weekly_plan": [
        {
            "day": "トレーニング曜日 (例: 月)",
            "menu": [
                {
                    "name": "種目名 (例: ベンチプレス)",
                    "sets": 3,
                    "reps": 5,
                    "weight": 80,
                    "is_max": false
                }
            ]
        }
    ]
}
"""

    prompt_context = f"""
あなたはベンチプレス専門のトレーニングコーチです。ユーザーの過去のパフォーマンスに基づいて、
最適化された次の週のトレーニングメニューを生成してください。

**ユーザー情報:**
- 現在の基準MAXベンチプレス: {max_bp}kg
- 目標ベンチプレス: {initial_info.get('goal_bp')}kg
- トレーニング頻度: 週{freq_next}回
- トレーニング曜日: {', '.join(weekdays)}

**トレーニングサイクルと目標:**
- 生成する週: {week_num}週目
- 4週サイクル: 1週目ボリューム / 2週目強度 / 3週目ピーク / 4週目MAX測定
- 4週目のMAX測定種目には 'is_max': true を設定し、reps=1, sets=1 にしてください。

**制約:**
- 重量は 2.5kg 単位
- 自重種目（腕立て・ディップス等）は使わず、バーベル・ダンベル・マシン等の外部負荷種目を使ってください。

**前週までの実績:**
{last_week_records}

以下の JSON スキーマに完全準拠する形で、{week_num}週目のメニューを生成してください。
{json_schema_str}
"""

    try:
        with st.spinner(f"🤖 AIが{week_num}週目のトレーニングメニューを生成中..."):
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはトレーニングメニューをJSON形式で出力するコーチです。",
                    },
                    {"role": "user", "content": prompt_context},
                ],
                response_format={"type": "json_object"},
            )
            response_json = json.loads(res.choices[0].message.content)
            return response_json.get("weekly_plan", [])
    except Exception as e:
        st.error(f"AIメニュー生成中にエラーが発生しました: {e}")
        return None


def generate_ai_daily_review(
    week_num: int,
    day_of_week: str,
    initial_info: Dict[str, Any],
    records: Dict[int, Dict[str, Any]],
    client: Any,
) -> str:
    """当日のトレーニング実績レビューを生成"""

    if client is None:
        return "AI機能が無効のためレビューできません。"

    current_records = format_records_for_prompt(
        records, week_num, initial_info["weekdays"], for_review=True, target_day=day_of_week
    )
    df_all = convert_records_to_dataframe(records)
    max_total_load = 0
    if not df_all.empty:
        bp_records = df_all[df_all["Exercise"].str.contains("ベンチプレス|BP", case=False, na=False)]
        if not bp_records.empty:
            max_total_load = int(
                bp_records.groupby("Week_Day_ID")["Total_Load"].sum().max()
            )

    prompt_context = f"""
あなたはユーザーの専属モチベーションコーチです。
以下の当日実績に基づき、2〜3文でポジティブなレビューを日本語で返してください。

**現在のMAXベンチプレス:** {st.session_state.max_test_result}kg
**目標ベンチプレス:** {initial_info.get('goal_bp')}kg
**これまでのベンチプレス系種目の最高総負荷量:** {max_total_load}kg

**レビュー対象の実績:**
{current_records}
"""

    try:
        with st.spinner(f"🧠 AIコーチが{day_of_week}のレビューを作成中..."):
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはトレーニング実績に基づいて短い励ましコメントを返す日本語コーチです。",
                    },
                    {"role": "user", "content": prompt_context},
                ],
            )
            return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AIレビュー生成中にエラーが発生しました: {e}")
        return "レビュー生成に失敗しました。"


def check_all_records_saved_for_day(
    week_num: int,
    day: str,
    week_plan: List[Dict[str, Any]],
    records: Dict[int, Dict[str, Any]],
):
    """その日のメニューが全部 records にあるか確認"""
    recorded_data = records.get(week_num, {}).get(day, {})

    for day_plan in week_plan:
        if day_plan["day"] == day:
            for item in day_plan["menu"]:
                name = item["name"]
                if name not in recorded_data:
                    return False, f"【{day}】{name}"
            return True, "完了"
    return False, "該当日がメニューに見つかりません"


def check_all_records_saved_for_week(
    week_num: int,
    week_plan: List[Dict[str, Any]],
    records: Dict[int, Dict[str, Any]],
):
    """週の全種目が入力されているか確認し、4週目ならMAX結果を自動設定"""
    recorded_data = records.get(week_num, {})
    max_measurement_done = False
    max_weight = 0

    for day_plan in week_plan:
        day = day_plan["day"]
        for item in day_plan["menu"]:
            name = item["name"]
            if day not in recorded_data or name not in recorded_data[day]:
                return False, f"【{day}】{name}"

            if week_num == 4 and item.get("is_max", False):
                max_measurement_done = True
                max_weight = recorded_data[day][name]["weight"]

    if week_num == 4 and max_measurement_done:
        st.session_state.max_test_result = int(max_weight)
        goal_bp = st.session_state.initial_info["goal_bp"]
        if st.session_state.max_test_result >= goal_bp:
            st.session_state.goal_achieved_pending = True
            st.session_state.max_registered_not_achieved = False
        else:
            st.session_state.goal_achieved_pending = False
            st.session_state.max_registered_not_achieved = True

    return True, "完了"


# =========================
# 汎用コメント（タンパク質など）
# =========================

def generate_feedback_message(client: Any, success_rate: float, context: str) -> str:
    if client is None:
        if success_rate >= 1.0:
            return "最高です！すべて達成できました！この調子でいきましょう💪"
        elif success_rate >= 0.7:
            return "かなり良いペースです！できた部分に自信を持って、次も一歩前進しましょう🔥"
        else:
            return "今回はうまくいかなかったところもあるかもしれませんが、その記録自体が大きな一歩です。継続していきましょう😊"

    prompt = f"""
あなたは筋トレコーチ兼メンタルトレーナーです。
ベンチプレス100kgを目指しているユーザーに対して、以下の結果に基づいて、
ポジティブで、優しく、前向きになれるコメントを日本語で1〜3文書いてください。

・対象: {context}
・達成率: {success_rate*100:.1f}%
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly Japanese strength coach."},
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content.strip()
    except Exception:
        if success_rate >= 1.0:
            return "最高です！すべて達成できました！この調子でいきましょう💪"
        elif success_rate >= 0.7:
            return "かなり良いペースです！できた部分に自信を持って、次も一歩前進しましょう🔥"
        else:
            return "今回はうまくいかなかったところもあるかもしれませんが、その記録自体が大きな一歩です。継続していきましょう😊"


def estimate_protein_from_image(client: Any, file) -> float:
    if client is None:
        st.warning("OpenAI API が設定されていないため、写真からの推定は行えません。手入力してください。")
        return 0.0

    img_bytes = file.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = """
この画像に写っている食事全体で、おおよそ何グラムのタンパク質が含まれているかを、
半角数字のみ（例: "25"）で出力してください。単位や説明は書かないでください。
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        text = res.choices[0].message.content.strip()
        grams = float(text.split()[0])
        return max(0.0, grams)
    except Exception:
        st.warning("画像からの推定に失敗しました。手入力を利用してください。")
        return 0.0


# =========================
# ページ1: 初期設定
# =========================

def page_initial_settings(client: Any):
    st.header("初期設定（プロフィール登録）")

    with st.form("initial_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("身長 (cm)", min_value=100.0, max_value=250.0, step=0.5)
            weight = st.number_input("体重 (kg)", min_value=30.0, max_value=200.0, step=0.5)
        with col2:
            current_1rm = st.number_input("現在のベンチプレス最高重量 (kg)", min_value=20.0, max_value=250.0, step=1.0)

        weekdays = st.multiselect(
            "今週ジムに行く曜日を選択",
            WEEKDAYS_JP,
            default=["月", "水", "金"],
        )

        submitted = st.form_submit_button("目標達成日と今週のメニューを作成")

    if submitted:
        if not weekdays:
            st.warning("少なくとも1日は選択してください。")
            return

        sessions_per_week = len(weekdays)
        weeks = estimate_weeks_to_target(current_1rm, sessions_per_week)
        today = date.today()
        target_date = today + timedelta(weeks=weeks)

        st.session_state.profile = {
            "height": height,
            "weight": weight,
            "current_1rm": current_1rm,
            "sessions_per_week": sessions_per_week,
            "target_weeks": weeks,
            "target_date": target_date.isoformat(),
            "start_date": today.isoformat(),
        }

        # タンパク質目標（体重×2g）
        st.session_state.protein_goal = weight * 2.0
        st.session_state.protein_today = 0.0

        # 100kgアプリ側のログ初期記録
        log_training_snapshot(note="初期設定", log_date=today)

        # 「今週のトレーニング」用の初期情報
        st.session_state.initial_info = {
            "height": height,
            "body_weight": weight,
            "current_bp": current_1rm,
            "goal_bp": TARGET_1RM,
            "freq": sessions_per_week,
            "weekdays": weekdays,
        }
        st.session_state.max_test_result = current_1rm

        # 1週目のAIメニュー生成
        week1_plan = generate_ai_week_plan(
            1,
            weekdays,
            st.session_state.initial_info,
            st.session_state.records,
            current_1rm,
            client,
        )
        if week1_plan:
            st.session_state.weekly_plan = [week1_plan]
            st.session_state.records = {}
            st.session_state.current_week = 1
            st.session_state.training_started = False
            st.session_state.goal_achieved_pending = False
            st.session_state.max_registered_not_achieved = False
            st.session_state.day_review_done = {}
            st.session_state.last_review = {}
            st.session_state.next_week_config_pending = False
            st.success("プロフィールと1週目のトレーニングメニューを作成しました！")
        else:
            st.error("AIメニュー生成に失敗しました。APIキー設定などを確認してください。")

        st.write(f"✅ ベンチプレス100kgまでの目安: **約 {weeks} 週間**")
        st.write(f"✅ 目標達成予定日: **{target_date} 頃**")

    if st.session_state.profile:
        st.subheader("現在登録されているプロフィール")
        p = st.session_state.profile
        st.write(f"- 身長: {p['height']} cm")
        st.write(f"- 体重: {p['weight']} kg")
        st.write(f"- 現在のベンチプレス最高重量: {p['current_1rm']} kg")
        st.write(f"- 今週ジムに行く曜日数: {p['sessions_per_week']} 日")
        st.write(f"- 100kgまでの目安: 約 {p['target_weeks']} 週間")
        st.write(f"- 目標達成予定日: {p['target_date']} 頃")


# =========================
# ページ2: 今週のトレーニング（元②トレーニング管理）
# =========================

def page_training_week(client: Any):
    st.header("今週のトレーニング")

    if not st.session_state.initial_info:
        st.info("まず「初期設定」タブでプロフィール・曜日などを登録してください。")
        return

    if client is None:
        st.error("OpenAI APIキーが設定されていません。AIメニュー生成・レビューは利用できません。")
        return

    # A. 次週設定画面
    if st.session_state.next_week_config_pending:
        st.subheader("⚙️ 次週トレーニング設定")

        is_cycle_restart = st.session_state.current_week == 0
        next_week_num = 1 if is_cycle_restart else st.session_state.current_week + 1

        if is_cycle_restart:
            st.warning(f"ベンチプレスサイクルが完了しました。次サイクル（{next_week_num}週目）の設定を行ってください。")
        else:
            st.warning(f"現在、{st.session_state.current_week}週目のトレーニングが全て完了しました。次週（{next_week_num}週目）の設定を行ってください。")

        with st.form("next_week_config_form"):
            default_freq = st.session_state.initial_info["freq"]
            default_weekdays = st.session_state.initial_info["weekdays"]

            freq_next = st.number_input(
                "次週のトレーニング回数", 1, 7, default_freq, key="freq_next_input"
            )
            weekdays_next = st.multiselect(
                "次週のトレーニング曜日",
                WEEKDAYS_JP,
                default=default_weekdays[:freq_next]
                if len(default_weekdays) != default_freq
                else default_weekdays,
                key="weekdays_next_select",
            )

            submit_config = st.form_submit_button("次週メニューを生成")

        if submit_config:
            if len(weekdays_next) != freq_next:
                st.error("次週回数と選択曜日の数が一致していません。")
                return

            # initial_info を更新
            st.session_state.initial_info["freq"] = freq_next
            st.session_state.initial_info["weekdays"] = weekdays_next

            max_to_use = st.session_state.max_test_result
            new_plan = generate_ai_week_plan(
                next_week_num,
                weekdays_next,
                st.session_state.initial_info,
                st.session_state.records,
                max_to_use,
                client,
            )

            if new_plan:
                st.session_state.current_week = next_week_num

                if is_cycle_restart:
                    st.session_state.weekly_plan = [new_plan]
                    st.session_state.records = {}
                else:
                    st.session_state.weekly_plan.append(new_plan)

                st.session_state.day_review_done = {}
                st.session_state.last_review = {}
                st.session_state.next_week_config_pending = False
                st.session_state.training_started = False
                st.session_state.goal_achieved_pending = False
                st.session_state.max_registered_not_achieved = False

                st.success(f"✅ {st.session_state.current_week}週目のAIメニュー生成完了。トレーニング開始ボタンを押してください。")
                st.rerun()
            else:
                st.error("AIメニュー生成に失敗しました。")
        return

    # B. 通常のトレーニング管理画面
    if not st.session_state.training_started:
        if st.button("🚀 トレーニング開始"):
            st.session_state.training_started = True
            st.success("トレーニング開始！下に実績入力フォームが表示されます。")
            st.rerun()
        return

    week_idx = st.session_state.current_week - 1
    week_number = st.session_state.current_week

    if week_idx >= len(st.session_state.weekly_plan):
        st.error("プランデータが見つかりません。初期設定から再生成してください。")
        st.session_state.training_started = False
        return

    week_plan = st.session_state.weekly_plan[week_idx]

    st.info(f"現在の基準MAX重量: **{st.session_state.max_test_result} kg**")

    st.subheader(f"📅 {week_number}週目のメニュー")

    for day_plan in week_plan:
        day = day_plan["day"]
        is_day_reviewed = st.session_state.day_review_done.get(week_number, {}).get(day, False)

        st.markdown(f"### {day} {'（レビュー完了）' if is_day_reviewed else ''}")

        is_day_fully_saved, missing_item_day = check_all_records_saved_for_day(
            week_number, day, week_plan, st.session_state.records
        )

        for item in day_plan["menu"]:
            name = item["name"]
            key_id = f"week{week_number}_{day}_{name}"

            saved_record = (
                st.session_state.records.get(week_number, {})
                .get(day, {})
                .get(name, None)
            )
            is_done = saved_record is not None
            is_disabled = is_day_reviewed

            expanded_state = not is_done and not is_disabled
            if key_id not in st.session_state.expanded_status:
                st.session_state.expanded_status[key_id] = expanded_state

            if week_number == 4 and item.get("is_max"):
                st.warning(
                    f"🚨 **{day}のベンチプレス**：今日はMAX測定日です。"
                    "**1回 × 1セット**で実績重量を入力してください。"
                )

            plan_info = f" ({item['weight']}kg x {item['reps']}回 x {item['sets']}セット 計画)"

            with st.expander(
                f"{name} {plan_info} {'✔ 保存済み' if is_done else ''}",
                expanded=st.session_state.expanded_status.get(key_id, expanded_state),
            ):
                w_val = int(saved_record["weight"]) if saved_record else int(item["weight"])
                r_val = int(saved_record["reps"]) if saved_record else int(item["reps"])
                s_val = int(saved_record["sets"]) if saved_record else int(item["sets"])
                note_val = saved_record["note"] if saved_record else ""

                cols = st.columns([1, 1, 1, 2, 1])
                w_input = cols[0].number_input(
                    "実績重量(kg)",
                    0,
                    500,
                    value=w_val,
                    key=f"{key_id}_w",
                    disabled=is_disabled,
                )

                if week_number == 4 and item.get("is_max"):
                    r_input = cols[1].number_input(
                        "実績回数",
                        1,
                        100,
                        value=r_val if r_val > 1 else 1,
                        key=f"{key_id}_r",
                        disabled=True,
                    )
                    s_input = cols[2].number_input(
                        "実績セット数",
                        1,
                        20,
                        value=s_val if s_val > 1 else 1,
                        key=f"{key_id}_s",
                        disabled=True,
                    )
                else:
                    r_input = cols[1].number_input(
                        "実績回数",
                        1,
                        100,
                        value=r_val,
                        key=f"{key_id}_r",
                        disabled=is_disabled,
                    )
                    s_input = cols[2].number_input(
                        "実績セット数",
                        1,
                        20,
                        value=s_val,
                        key=f"{key_id}_s",
                        disabled=is_disabled,
                    )

                note_input = cols[3].text_input(
                    "メモ",
                    value=note_val,
                    key=f"{key_id}_note",
                    disabled=is_disabled,
                )
                save_btn = cols[4].button(
                    "保存", key=f"{key_id}_save", disabled=is_disabled
                )

                if save_btn:
                    if week_number not in st.session_state.records:
                        st.session_state.records[week_number] = {}
                    if day not in st.session_state.records[week_number]:
                        st.session_state.records[week_number][day] = {}

                    if week_number == 4 and item.get("is_max"):
                        final_reps = 1
                        final_sets = 1
                    else:
                        final_reps = int(r_input)
                        final_sets = int(s_input)

                    st.session_state.records[week_number][day][name] = {
                        "weight": int(w_input),
                        "reps": final_reps,
                        "sets": final_sets,
                        "note": note_input,
                        "is_max": item.get("is_max", False),
                    }
                    st.session_state.expanded_status[key_id] = False
                    st.success(f"{name} の実績を保存しました！")
                    st.rerun()

        # 曜日ごとのレビュー生成
        if is_day_fully_saved and not is_day_reviewed:
            if st.button(f"🗓️ {day}のトレーニング終了！レビューを見る", key=f"finish_day_btn_{day}"):
                review_text = generate_ai_daily_review(
                    week_number,
                    day,
                    st.session_state.initial_info,
                    st.session_state.records,
                    client,
                )
                st.session_state.last_review.setdefault(week_number, {})[day] = review_text
                st.session_state.day_review_done.setdefault(week_number, {})[day] = True
                st.success(f"{day}のレビューが完了しました。")
                st.rerun()

        if is_day_fully_saved and is_day_reviewed:
            st.subheader(f"🎉 {day} AIコーチングレビュー")
            with st.container(border=True):
                st.markdown(
                    st.session_state.last_review.get(week_number, {}).get(day, "レビュー生成待ち...")
                )
            st.markdown("---")
        elif is_day_fully_saved and not is_day_reviewed:
            st.info("全ての種目を記録しました。レビューボタンを押して評価を確認しましょう。")
        elif not is_day_fully_saved:
            st.warning(f"⚠️ {day}にはまだ未完了の種目 ({missing_item_day}) があります。")

    st.markdown("---")

    # 週完了判定
    is_week_fully_recorded, _ = check_all_records_saved_for_week(
        week_number, week_plan, st.session_state.records
    )
    all_days_reviewed = all(
        st.session_state.day_review_done.get(week_number, {}).get(dp["day"], False)
        for dp in week_plan
    )

    if is_week_fully_recorded and all_days_reviewed:
        # 4週目完了
        if week_number == 4:
            max_result = st.session_state.max_test_result
            goal = st.session_state.initial_info["goal_bp"]

            if st.session_state.goal_achieved_pending:
                st.balloons()
                st.success(f"🎉 目標達成おめでとうございます！ ({max_result}kg達成 / 目標 {goal}kg)")

                new_goal = st.number_input(
                    "次の目標ベンチプレス重量を入力してください",
                    min_value=max_result,
                    max_value=300,
                    value=int(max_result + 5),
                    key="new_goal_input",
                )
                if st.button("新目標でAIサイクル再スタート", key="restart_new_goal_btn"):
                    st.session_state.initial_info["goal_bp"] = int(new_goal)
                    st.session_state.current_week = 0
                    st.session_state.next_week_config_pending = True
                    st.rerun()
            elif st.session_state.max_registered_not_achieved:
                st.info(
                    f"目標未達です。({max_result}kg / 目標 {goal}kg) 次のサイクルで必ず達成しましょう！"
                )
                if st.button("目標据え置きでAIサイクル再スタート", key="restart_same_goal"):
                    st.session_state.current_week = 0
                    st.session_state.next_week_config_pending = True
                    st.rerun()
        # 1〜3週目完了
        elif week_number < 4:
            if st.button("次週のトレーニング設定とAIメニュー生成へ", key="next_week_config_btn"):
                st.session_state.next_week_config_pending = True
                st.rerun()
    elif is_week_fully_recorded and not all_days_reviewed:
        st.warning("⚠️ 未完了のレビューがあります。全ての日のレビューを完了すると次週へ進めます。")


# =========================
# ページ3: タンパク質管理
# =========================

def page_protein(client: Any):
    st.header("1日のタンパク質管理")

    if not st.session_state.profile:
        st.info("まずは「初期設定」でプロフィールを登録してください。")
        return

    weight = st.session_state.profile["weight"]
    if st.session_state.protein_goal <= 0:
        st.session_state.protein_goal = weight * 2.0

    goal = st.session_state.protein_goal
    today = date.today().isoformat()

    st.write(f"本日の推奨タンパク質量の目安: **{goal:.0f} g** （体重 {weight}kg × 2g）")
    st.write(f"今日の日付: {today}")

    st.subheader("食事ごとのタンパク質量を入力")
    col1, col2 = st.columns(2)

    with col1:
        manual_amount = st.number_input(
            "手入力でタンパク質量を追加 (g)",
            min_value=0.0,
            step=1.0,
            key="manual_protein",
        )
        if st.button("この量を追加"):
            st.session_state.protein_today += manual_amount
            st.success(f"{manual_amount:.1f} g を追加しました！")

    with col2:
        st.write("食事の写真からざっくり推定（ChatGPT）")
        img_file = st.file_uploader(
            "食事の写真をアップロード", type=["jpg", "jpeg", "png"], key="protein_image"
        )
        if img_file is not None:
            if st.button("写真からタンパク質量を推定"):
                grams = estimate_protein_from_image(client, img_file)
                if grams > 0:
                    st.session_state.protein_today += grams
                    st.success(f"推定 {grams:.1f} g を追加しました！")

    # ゲージ表示
    st.markdown("---")
    st.subheader("本日の達成状況")

    consumed = st.session_state.protein_today
    ratio = min(consumed / goal, 1.0) if goal > 0 else 0.0

    st.write(f"本日摂取量: **{consumed:.1f} g / {goal:.1f} g**")
    st.progress(ratio)

    if consumed >= goal and goal > 0:
        if st.session_state.protein_celebrated_date != today:
            st.balloons()
            st.session_state.protein_celebrated_date = today

        msg = generate_feedback_message(client, 1.0, "本日のタンパク質摂取")
        st.success("目標達成おめでとうございます！🎉")
        st.markdown("**コーチからのコメント**")
        st.write(msg)

    if st.button("今日の記録をリセット（テスト用）"):
        st.session_state.protein_today = 0.0
        st.success("本日のタンパク質記録をリセットしました。")


# =========================
# ページ4: 進捗・ロードマップ / MAXテスト
# =========================

def page_progress_and_roadmap():
    st.header("進捗・ロードマップ / MAXテスト")

    if not st.session_state.profile:
        st.info("まずは「初期設定」でプロフィールを登録してください。")
        return

    profile = st.session_state.profile
    current_1rm = float(profile.get("current_1rm", 0.0))

    # 100kg達成お祝い
    if current_1rm >= TARGET_1RM:
        if not st.session_state.celebrated_100kg:
            st.balloons()
            st.session_state.celebrated_100kg = True

        achieved_date = get_first_100kg_date()
        date_text = achieved_date if achieved_date else "（日付記録なし）"

        st.markdown("## 🏆 100kg Club 認定 🏆")
        st.markdown(
            f"""
            <div style="
                border-radius: 16px;
                border: 2px solid #f97316;
                padding: 16px 20px;
                background: radial-gradient(circle at top, #111827, #020617);
                text-align: center;
            ">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">Congratulations!</div>
                <div style="font-size: 2.0rem; font-weight: bold; margin-bottom: 8px;">
                    あなたは <span style="color:#f97316;">100kg Club</span> のメンバーです
                </div>
                <div style="font-size: 1rem; color:#e5e7eb;">
                    達成日：<span style="font-weight:bold;">{date_text}</span><br/>
                    記録：<span style="font-weight:bold;">{current_1rm:.1f} kg</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

    # 100kgまでの進捗バー
    st.subheader("100kgまでの進捗")
    progress_ratio = min(current_1rm / TARGET_1RM, 1.0) if TARGET_1RM > 0 else 0.0
    st.write(f"現在のベンチプレス最高重量: **{current_1rm:.1f} kg / {TARGET_1RM} kg**")
    st.progress(progress_ratio)

    # 推定1RMの理想カーブ vs 実測
    st.subheader("推定1RMの推移（実測 vs 目標ペース）")

    logs_sorted = sorted(st.session_state.training_logs, key=lambda x: x["date"]) if st.session_state.training_logs else []

    if logs_sorted:
        try:
            start_date = date.fromisoformat(profile.get("start_date"))
            target_date = date.fromisoformat(profile.get("target_date"))
        except Exception:
            start_date = date.fromisoformat(logs_sorted[0]["date"])
            target_date = start_date + timedelta(weeks=profile.get("target_weeks", 12))

        if target_date <= start_date:
            target_date = start_date + timedelta(days=1)

        initial_1rm = float(logs_sorted[0]["current_1rm"])

        num_days = (target_date - start_date).days
        if num_days < 1:
            num_days = 1

        ideal_dates = [start_date + timedelta(days=i) for i in range(num_days + 1)]
        ideal_values = [
            initial_1rm + (TARGET_1RM - initial_1rm) * (i / num_days)
            for i in range(num_days + 1)
        ]

        df_ideal = pd.DataFrame({"date": ideal_dates, "目標ペース1RM(kg)": ideal_values})

        actual_dates = [date.fromisoformat(log["date"]) for log in logs_sorted]
        actual_values = [float(log["current_1rm"]) for log in logs_sorted]
        df_actual = pd.DataFrame({"date": actual_dates, "実測1RM(kg)": actual_values})

        # 同日付をまとめて最大値を採用
        df_actual = df_actual.sort_values("date").groupby("date", as_index=False)["実測1RM(kg)"].max()

        max_actual = df_actual["実測1RM(kg)"].max()
        y_max = max(TARGET_1RM, max_actual)
        y_min = initial_1rm

        base_ideal = alt.Chart(df_ideal).encode(x=alt.X("date:T", title="日付"))
        base_actual = alt.Chart(df_actual).encode(x=alt.X("date:T", title="日付"))

        ideal_line = base_ideal.mark_line().encode(
            y=alt.Y(
                "目標ペース1RM(kg):Q",
                title="ベンチプレス1RM (kg)",
                scale=alt.Scale(domain=[y_min, y_max]),
            ),
            color=alt.value("#f97316"),
            tooltip=["date:T", "目標ペース1RM(kg):Q"],
        )

        actual_line = base_actual.mark_line(point=True).encode(
            y=alt.Y(
                "実測1RM(kg):Q",
                title="ベンチプレス1RM (kg)",
                scale=alt.Scale(domain=[y_min, y_max]),
            ),
            color=alt.value("#38bdf8"),
            tooltip=["date:T", "実測1RM(kg):Q"],
        )

        chart = alt.layer(ideal_line, actual_line).resolve_scale(y="shared").properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("まだ推移グラフ用のログがありません。初期設定やMAXテストを行うと記録されます。")

    st.markdown("---")

    # ロードマップ
    st.subheader("100kgロードマップ（中間ゴール）")
    milestones = [
        (60, "ビギナー脱出"),
        (70, "フォームを固めるフェーズ"),
        (80, "中級レベルへの入口"),
        (90, "90kgチャレンジャー"),
        (95, "100kg目前ゾーン"),
        (100, "100kgクラブ"),
    ]
    for kg, label in milestones:
        reached = current_1rm >= kg
        status = "✅" if reached else "⬜"
        st.markdown(f"{status} **{kg}kg** : {label}")

    st.markdown("---")

    # MAXテスト
    st.subheader("MAXテスト（1RMテスト）")

    with st.form("max_test_form"):
        test_date = st.date_input("MAXテストを実施した日", value=date.today())
        test_1rm = st.number_input(
            "今回テストで挙がった最大重量（1回挙上できた重量）(kg)",
            min_value=0.0,
            max_value=300.0,
            step=1.0,
        )
        submitted = st.form_submit_button("MAXテスト結果を登録")

    if submitted:
        if test_1rm <= 0:
            st.warning("1回挙がった重量を入力してください。")
        else:
            st.session_state.profile["current_1rm"] = float(test_1rm)
            log_training_snapshot(note="MAXテストで更新", log_date=test_date)
            st.success(f"MAXテスト結果を登録しました！ 現在のベンチプレス最高重量: {test_1rm:.1f} kg")
            st.rerun()


# =========================
# メイン
# =========================

def main():
    st.set_page_config(page_title="ベンチプレス100kgアプリ", page_icon="💪", layout="wide")
    init_state()
    client = get_openai_client()

    # ===== ダークテーマ＋見やすい文字色 =====
    st.markdown(
        """
    <style>
    /* 全体背景をダーク */
    [data-testid="stAppViewContainer"] {
        background-color: #020617;
    }
    [data-testid="stSidebar"] {
        background-color: #020617;
    }

    /* 見出しは明るい色 */
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB !important;
    }

    /* 通常テキスト・説明文も明るい色（黒背景上） */
    .stMarkdown, .stText, .stCaption, .stSubheader, .stCheckbox, .stRadio, label {
        color: #E5E7EB !important;
    }

    /* 入力欄の中の文字は黒で見やすく */
    input, textarea, select,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox div[data-baseweb="select"] input {
        color: #111827 !important;
    }

    /* ボタンをオレンジ系に */
    .stButton>button {
        background-color: #f97316;
        color: white;
        border-radius: 999px;
        border: 1px solid #ea580c;
    }
    .stButton>button:hover {
        background-color: #ea580c;
        border-color: #c2410c;
        color: white;
    }

    /* プログレスバーの色 */
    .stProgress > div > div {
        background-color: #f97316;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if client is None:
        st.warning("OpenAI APIキーが未設定です。AIメニュー生成・レビュー機能が使えません。")

    st.title("💪 ベンチプレス100kgチャレンジアプリ")
    st.caption("Streamlit × ChatGPT 版（Supabase連携はこれから）")

    tabs = st.tabs(
        [
            "初期設定",
            "今週のトレーニング",
            "タンパク質管理",
            "進捗・ロードマップ / テスト",
        ]
    )

    with tabs[0]:
        page_initial_settings(client)
    with tabs[1]:
        page_training_week(client)
    with tabs[2]:
        page_protein(client)
    with tabs[3]:
        page_progress_and_roadmap()


if __name__ == "__main__":
    main()

# =========================
# supabase連携
# =========================

#Supabaseクライアントの初期化
from supabase import create_client, Client

def get_supabase_client() -> Optional[Client]:
    try:
        if "supabase" not in st.secrets:
            return None
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        # secrets.tomlが存在しない、またはSupabase設定がない場合は無視
        return None

#PythonコードにSupabaseクライアントを追加
supabase = get_supabase_client()

#保存・取得処理をSupabaseに置き換える
def log_training_snapshot(note: str = "", log_date: Optional[date] = None, supabase: Optional[Client] = None):
    if not st.session_state.profile or supabase is None:
        return

    d = log_date.isoformat() if log_date else date.today().isoformat()
    log = {
        "date": d,
        "current_1rm": float(st.session_state.profile.get("current_1rm", 0.0)),
        "note": note,
    }

    try:
        supabase.table("training_logs").insert(log).execute()
    except Exception as e:
        st.error(f"Supabaseへの保存に失敗しました: {e}")

