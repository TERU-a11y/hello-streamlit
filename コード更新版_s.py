# app.py
# ベンチプレス100kgチャレンジアプリ
# Streamlit + ChatGPT（OpenAI）版 / Supabase連携はまだナシ

import streamlit as st
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
import base64
import pandas as pd  # 推定1RMグラフ用
import altair as alt  # グラフカスタマイズ用

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================
# 定数
# =========================

TARGET_1RM = 100  # 目標ベンチプレス 100kg


# =========================
# セッションステート初期化
# =========================

def init_state():
    if "profile" not in st.session_state:
        st.session_state.profile: Dict[str, Any] = {}
    if "week_plan" not in st.session_state:
        st.session_state.week_plan: List[Dict[str, Any]] = []  # 今週メニュー
    if "workout_done" not in st.session_state:
        st.session_state.workout_done: Dict[str, bool] = {}  # メニューの達成状況
    if "protein_goal" not in st.session_state:
        st.session_state.protein_goal: float = 0.0
    if "protein_today" not in st.session_state:
        st.session_state.protein_today: float = 0.0

    # 何週目かを管理（0: 1週目）
    if "current_week_index" not in st.session_state:
        st.session_state.current_week_index: int = 0
    # 「今週の結果を振り返る」表示フラグ
    if "show_week_summary" not in st.session_state:
        st.session_state.show_week_summary: bool = False

    # トレーニング進捗ログ（1RM推移など）
    if "training_logs" not in st.session_state:
        st.session_state.training_logs: List[Dict[str, Any]] = []

    # 100kg達成イベントをすでに祝ったかどうか
    if "celebrated_100kg" not in st.session_state:
        st.session_state.celebrated_100kg: bool = False

    # タンパク質目標を祝った日付（同じ日に何度もバルーンが出ないように）
    if "protein_celebrated_date" not in st.session_state:
        st.session_state.protein_celebrated_date: Optional[str] = None

    # その週のトレーニング100%達成を祝ったかどうか
    if "celebrated_this_week" not in st.session_state:
        st.session_state.celebrated_this_week: bool = False


# =========================
# OpenAI クライアント
# =========================

def get_openai_client() -> Any:
    if OpenAI is None:
        return None
    try:
        api_key = st.secrets["openai"]["api_key"]
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


# =========================
# ロジック: トレーニング関連
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


def generate_week_plan(current_1rm: float, sessions_per_week: int, week_index: int) -> List[Dict[str, Any]]:
    """
    1週間分のメニューを生成（ベンチ＋補助種目込み）
    各種目に「どの部位に効くか（muscles）」も付与
    """
    progression = 1.0 + week_index * 0.02  # 週ごとに+2%
    base_1rm = current_1rm * progression

    plan: List[Dict[str, Any]] = []

    for session in range(1, sessions_per_week + 1):
        session_label = f"{week_index+1}週目・{session}回目"

        main_weight = round(base_1rm * 0.8)   # メインセット
        vol_weight = round(base_1rm * 0.7)    # ボリューム
        tech_weight = round(base_1rm * 0.6)   # テクニック
        row_weight = round(base_1rm * 0.6)
        ohp_weight = round(base_1rm * 0.5)

        items_for_session = [
            {
                "id": f"w{week_index}_s{session}_bench_main",
                "session_label": session_label,
                "name": "ベンチプレス（メインセット）",
                "detail": f"{main_weight}kg x 5回",
                "muscles": "大胸筋・三角筋前部・上腕三頭筋・体幹・下半身（脚）",
            },
            {
                "id": f"w{week_index}_s{session}_bench_vol",
                "session_label": session_label,
                "name": "ベンチプレス（ボリュームセット）",
                "detail": f"{vol_weight}kg x 8回 × 3セット",
                "muscles": "大胸筋・三角筋前部・上腕三頭筋・体幹・下半身（脚）",
            },
            {
                "id": f"w{week_index}_s{session}_bench_tech",
                "session_label": session_label,
                "name": "ベンチプレス（フォーム練習）",
                "detail": f"{tech_weight}kg x 10回 × 2セット",
                "muscles": "大胸筋・三角筋前部・上腕三頭筋・体幹",
            },
            {
                "id": f"w{week_index}_s{session}_incline",
                "session_label": session_label,
                "name": "インクラインダンベルプレス",
                "detail": "RPE 7〜8 で 8〜10回 × 3セット",
                "muscles": "大胸筋上部・三角筋前部・上腕三頭筋",
            },
            {
                "id": f"w{week_index}_s{session}_row",
                "session_label": session_label,
                "name": "バーベル（またはダンベル）ローイング",
                "detail": f"{row_weight}kg 相当 x 8〜10回 × 3セット",
                "muscles": "広背筋・僧帽筋・三角筋後部・体幹",
            },
            {
                "id": f"w{week_index}_s{session}_ohp",
                "session_label": session_label,
                "name": "ショルダープレス（バーベル or ダンベル）",
                "detail": f"{ohp_weight}kg 相当 x 6〜8回 × 3セット",
                "muscles": "三角筋前部・側部・上腕三頭筋・体幹",
            },
            {
                "id": f"w{week_index}_s{session}_triceps",
                "session_label": session_label,
                "name": "ディップス / トライセプスエクステンション",
                "detail": "自重 or 軽負荷で10〜12回 × 3セット",
                "muscles": "上腕三頭筋・大胸筋下部・前腕",
            },
            {
                "id": f"w{week_index}_s{session}_pushup",
                "session_label": session_label,
                "name": "プッシュアップ（腕立て伏せ）",
                "detail": "限界 -2回 を目安に 2〜3セット",
                "muscles": "大胸筋・三角筋前部・上腕三頭筋・体幹",
            },
            {
                "id": f"w{week_index}_s{session}_core",
                "session_label": session_label,
                "name": "フロントプランク",
                "detail": "40〜60秒 × 2〜3セット",
                "muscles": "体幹（腹直筋・腹横筋・腹斜筋・脊柱起立筋）",
            },
            {
                "id": f"w{week_index}_s{session}_legs",
                "session_label": session_label,
                "name": "ブルガリアンスクワット",
                "detail": "左右各 8〜10回 × 2〜3セット",
                "muscles": "下半身（大腿四頭筋・ハムストリングス・臀筋）",
            },
        ]

        plan.extend(items_for_session)

    return plan


def log_training_snapshot(note: str = "", log_date: Optional[date] = None):
    """現在の1RM等を training_logs に記録（log_date を指定できるように）"""
    if not st.session_state.profile:
        return

    if log_date is None:
        d = date.today().isoformat()
    else:
        if isinstance(log_date, date):
            d = log_date.isoformat()
        else:
            d = str(log_date)

    log = {
        "date": d,
        "week_index": st.session_state.current_week_index,
        "current_1rm": float(st.session_state.profile.get("current_1rm", 0.0)),
        "note": note,
    }
    st.session_state.training_logs.append(log)


def get_first_100kg_date() -> Optional[str]:
    """初めて current_1rm が 100kg に到達/超えた日付を返す"""
    logs = st.session_state.training_logs
    if not logs:
        return None
    sorted_logs = sorted(logs, key=lambda x: x["date"])
    for log in sorted_logs:
        if float(log.get("current_1rm", 0.0)) >= TARGET_1RM:
            return log["date"]
    return None


# =========================
# ロジック: ChatGPT コメント生成
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

条件:
- タメ口ではなく、丁寧語（です・ます）で話す
- 説教はしない
- できた点・続けた点を必ず褒める
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
    """食事写真からタンパク質量をざっくり推定（ChatGPT Vision想定）"""
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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
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
# UI: 初期設定
# =========================

def page_initial_settings():
    st.header("初期設定（プロフィール登録）")

    with st.form("initial_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("身長 (cm)", min_value=100.0, max_value=250.0, step=0.5)
            weight = st.number_input("体重 (kg)", min_value=30.0, max_value=200.0, step=0.5)
        with col2:
            current_1rm = st.number_input("現在のベンチプレス最高重量 (kg)", min_value=20.0, max_value=200.0, step=1.0)
            sessions_per_week = st.number_input("今週ジムに行ける回数", min_value=1, max_value=7, step=1)

        submitted = st.form_submit_button("目標達成日と今週のメニューを作成")
        if submitted:
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

            # 1週目のトレーニングメニュー生成（index 0）
            st.session_state.current_week_index = 0
            week_plan = generate_week_plan(current_1rm, sessions_per_week, week_index=0)
            st.session_state.week_plan = week_plan
            st.session_state.workout_done = {}
            st.session_state.show_week_summary = False
            st.session_state.celebrated_this_week = False  # その週の祝福フラグをリセット

            # タンパク質推奨量をセット（体重×2g）
            st.session_state.protein_goal = weight * 2.0
            st.session_state.protein_today = 0.0

            # トレーニングログ初期記録
            log_training_snapshot(note="初期設定", log_date=today)

            st.success("プロフィールと今週のトレーニングメニューを作成しました！")
            st.write(f"✅ ベンチプレス100kgまでの目安: **約 {weeks} 週間**")
            st.write(f"✅ 目標達成予定日: **{target_date} 頃**")

    if st.session_state.profile:
        st.subheader("現在登録されているプロフィール")
        p = st.session_state.profile
        st.write(f"- 身長: {p['height']} cm")
        st.write(f"- 体重: {p['weight']} kg")
        st.write(f"- 現在のベンチプレス最高重量: {p['current_1rm']} kg")
        st.write(f"- 今週ジムに行ける回数: {p['sessions_per_week']} 回")
        st.write(f"- 100kgまでの目安: 約 {p['target_weeks']} 週間")
        st.write(f"- 目標達成予定日: {p['target_date']} 頃")


# =========================
# UI: 今週のトレーニング
# =========================

def page_training_week(client: Any):
    st.header("今週のトレーニング")

    if not st.session_state.profile:
        st.info("まずは「初期設定」でプロフィールを登録してください。")
        return

    week_plan = st.session_state.week_plan
    if not week_plan:
        st.info("今週のトレーニングメニューがまだ生成されていません。「初期設定」で作成してください。")
        return

    current_week = st.session_state.current_week_index + 1
    st.caption(f"表示中: {current_week}週目のメニュー")

    st.subheader("今週のトレーニングメニュー（チェック式）")

    # セッションごとにグルーピング
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in week_plan:
        grouped.setdefault(item["session_label"], []).append(item)

    total = len(week_plan)

    # 表風にチェックボックス＋種目名＋詳細を表示（カード風ボックス付き）
    for session_label in sorted(grouped.keys()):
        # カードの開始
        st.markdown("<div class='workout-card'>", unsafe_allow_html=True)

        st.markdown(f"### {session_label}")
        header_cols = st.columns([0.1, 0.4, 0.5])
        with header_cols[0]:
            st.markdown("**完了**")
        with header_cols[1]:
            st.markdown("**種目**")
        with header_cols[2]:
            st.markdown("**内容 / 対象筋**")

        for item in grouped[session_label]:
            item_id = item["id"]
            if item_id not in st.session_state.workout_done:
                st.session_state.workout_done[item_id] = False

            cols = st.columns([0.1, 0.4, 0.5])
            with cols[0]:
                cb_key = f"chk_{st.session_state.current_week_index}_{item_id}"
                checked = st.checkbox(
                    "",
                    key=cb_key,
                    value=st.session_state.workout_done[item_id],
                )
                st.session_state.workout_done[item_id] = checked

            name_text = item["name"]
            detail_text = item["detail"]
            muscles_text = item.get("muscles", "")

            # チェック有無でスタイル切り替え
            if st.session_state.workout_done[item_id]:
                name_html = (
                    f"<span style='color: #9CA3AF; text-decoration: line-through;'>{name_text}</span>"
                )
                detail_html = (
                    f"<span style='color: #9CA3AF; text-decoration: line-through;'>{detail_text}</span>"
                )
                if muscles_text:
                    detail_html += (
                        f"<br/><span style='color:#6B7280; text-decoration: line-through; font-size:0.8rem;'>"
                        f"対象筋: {muscles_text}</span>"
                    )
            else:
                name_html = f"<span>{name_text}</span>"
                detail_html = f"<span>{detail_text}</span>"
                if muscles_text:
                    detail_html += (
                        f"<br/><span style='color:#9CA3AF; font-size:0.8rem;'>"
                        f"対象筋: {muscles_text}</span>"
                    )

            with cols[1]:
                st.markdown(name_html, unsafe_allow_html=True)
            with cols[2]:
                st.markdown(detail_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

    done_count = sum(1 for item in week_plan if st.session_state.workout_done.get(item["id"], False))
    success_rate = done_count / total if total > 0 else 0

    st.write(f"今週の達成状況: **{done_count} / {total} メニュー**（達成率 {success_rate*100:.1f}%）")

    # ---- 今週を締める ----
    st.subheader("今週のトレーニングを締める")
    st.write("今週のトレーニングが一通り終わったタイミングで押してください。")

    if st.button("今週の結果を振り返る"):
        st.session_state.show_week_summary = True

    if st.session_state.show_week_summary:
        context = "1週間のトレーニング"
        msg = generate_feedback_message(client, success_rate, context)

        # 100%達成ならバルーン（週につき1回だけ）
        if success_rate >= 1.0 and not st.session_state.celebrated_this_week:
            st.balloons()
            st.session_state.celebrated_this_week = True

        if success_rate >= 1.0:
            st.success("すべてのメニュー達成おめでとうございます！🎉")
        elif success_rate >= 0.7:
            st.info("かなり良い達成度です！🔥")
        else:
            st.info("記録を残せただけでも大きな一歩です😊")

        st.markdown("**コーチからのコメント**")
        st.write(msg)

        # 来週のジム回数入力
        st.markdown("---")
        st.subheader("来週のジム回数を入力してメニューを生成")
        next_sessions = st.number_input(
            "来週ジムに行ける回数",
            min_value=1,
            max_value=7,
            step=1,
            key="next_sessions_per_week"
        )

        if st.button("来週のメニューを作成"):
            current_1rm = float(st.session_state.profile["current_1rm"])

            # 達成度に応じて少し成長（シンプル版）
            new_1rm_for_profile = current_1rm * (1 + success_rate * 0.02)
            training_1rm_for_plan = new_1rm_for_profile
            note = "通常週"

            # プロフィール更新
            st.session_state.profile["current_1rm"] = round(new_1rm_for_profile, 1)
            st.session_state.profile["sessions_per_week"] = next_sessions

            # 週番号を1つ進める
            st.session_state.current_week_index += 1
            next_week_index = st.session_state.current_week_index

            # 新しい1週間分のメニューを生成
            new_week_plan = generate_week_plan(training_1rm_for_plan, next_sessions, week_index=next_week_index)

            st.session_state.week_plan = new_week_plan
            st.session_state.workout_done = {}  # チェックリセット
            st.session_state.show_week_summary = False  # 振り返り表示を閉じる
            st.session_state.celebrated_this_week = False  # 次の週用にリセット

            # トレーニングログに記録（今日の日付でOK）
            log_training_snapshot(note=note)

            st.success(f"{next_week_index+1}週目のトレーニングメニューを作成しました！")
            st.rerun()


# =========================
# UI: タンパク質管理
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
        manual_amount = st.number_input("手入力でタンパク質量を追加 (g)", min_value=0.0, step=1.0, key="manual_protein")
        if st.button("この量を追加"):
            st.session_state.protein_today += manual_amount
            st.success(f"{manual_amount:.1f} g を追加しました！")

    with col2:
        st.write("食事の写真からざっくり推定（ChatGPT）")
        img_file = st.file_uploader("食事の写真をアップロード", type=["jpg", "jpeg", "png"], key="protein_image")
        if img_file is not None:
            if st.button("写真からタンパク質量を推定"):
                grams = estimate_protein_from_image(client, img_file)
                if grams > 0:
                    st.session_state.protein_today += grams
                    st.success(f"推定 {grams:.1f} g を追加しました！")

    # ゲージ（progressバー）表示
    st.markdown("---")
    st.subheader("本日の達成状況")

    consumed = st.session_state.protein_today
    ratio = min(consumed / goal, 1.0) if goal > 0 else 0.0

    st.write(f"本日摂取量: **{consumed:.1f} g / {goal:.1f} g**")
    st.progress(ratio)

    # コメント & アニメーション（目標達成時のみ）
    if consumed >= goal and goal > 0:
        # 今日まだ祝ってなければバルーン
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
# UI: 進捗・ロードマップ / MAXテスト
# =========================

def page_progress_and_roadmap():
    st.header("進捗・ロードマップ / MAXテスト")

    if not st.session_state.profile:
        st.info("まずは「初期設定」でプロフィールを登録してください。")
        return

    current_1rm = float(st.session_state.profile.get("current_1rm", 0.0))

    # ---- 100kg達成のお祝い（初回のみ）----
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
                background: radial-gradient(circle at top, #1f2937, #020617);
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
                <div style="margin-top: 10px; font-size:0.95rem; color:#9ca3af;">
                    ここからは「維持」と「余裕を持って100kgを扱えること」を一緒に目指していきましょう。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

    # ---- 100kgまでの進捗バー ----
    st.subheader("100kgまでの進捗")
    progress_ratio = min(current_1rm / TARGET_1RM, 1.0) if TARGET_1RM > 0 else 0.0
    st.write(f"現在のベンチプレス最高重量: **{current_1rm:.1f} kg / {TARGET_1RM} kg**")
    st.progress(progress_ratio)

    # ---- 推定1RM推移グラフ（実測 + 目標ペース ＝ 理想カーブ） ----
    st.subheader("推定1RMの推移（実測 vs 目標ペース）")

    if st.session_state.training_logs:
        logs_sorted = sorted(st.session_state.training_logs, key=lambda x: x["date"])

        profile = st.session_state.profile

        # 開始日・目標日
        try:
            start_date = date.fromisoformat(profile.get("start_date"))
            target_date = date.fromisoformat(profile.get("target_date"))
        except Exception:
            start_date = date.fromisoformat(logs_sorted[0]["date"])
            target_date = start_date + timedelta(weeks=profile.get("target_weeks", 12))

        if target_date <= start_date:
            target_date = start_date + timedelta(days=1)

        # 初期の1RM（理想カーブのスタート）＝ 初期設定時の値
        initial_1rm = float(logs_sorted[0]["current_1rm"])

        # 目標日まで毎日1点の理想カーブを作る
        num_days = (target_date - start_date).days
        if num_days < 1:
            num_days = 1

        ideal_dates = [start_date + timedelta(days=i) for i in range(num_days + 1)]
        ideal_values = []
        for i, _d in enumerate(ideal_dates):
            frac = i / num_days  # 0〜1
            ideal_1rm = initial_1rm + (TARGET_1RM - initial_1rm) * frac
            ideal_values.append(ideal_1rm)

        df_ideal = pd.DataFrame({
            "date": ideal_dates,
            "目標ペース1RM(kg)": ideal_values,
        })

        # 実測ログ（training_logs）
        actual_dates = [date.fromisoformat(log["date"]) for log in logs_sorted]
        actual_values = [float(log["current_1rm"]) for log in logs_sorted]
        df_actual = pd.DataFrame({
            "date": actual_dates,
            "実測1RM(kg)": actual_values,
        })

        # 同じ日付が複数ある場合は1つにまとめる（最大値を採用）
        df_actual = (
            df_actual
            .sort_values("date")
            .groupby("date", as_index=False)["実測1RM(kg)"]
            .max()
        )

        # 実測値を「日次」に補完して線でつなぐ（前の値を維持）
        min_actual_date = df_actual["date"].min()
        max_actual_date = df_actual["date"].max()
        full_index = pd.date_range(min_actual_date, max_actual_date, freq="D")

        df_actual_dense = (
            df_actual.set_index("date")
            .reindex(full_index)
            .sort_index()
            .ffill()
            .reset_index()
        )
        df_actual_dense.rename(columns={"index": "date"}, inplace=True)

        # Y軸は「初期設定の1RM〜max(100kg, 実測最大値)」
        max_actual = df_actual["実測1RM(kg)"].max() if not df_actual.empty else initial_1rm
        y_max = max(TARGET_1RM, max_actual)
        y_min = initial_1rm  # 0kgではなく初期1RMからスタート

        base_ideal = alt.Chart(df_ideal).encode(
            x=alt.X("date:T", title="日付")
        )

        base_actual = alt.Chart(df_actual_dense).encode(
            x=alt.X("date:T", title="日付")
        )

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

        chart = alt.layer(ideal_line, actual_line).resolve_scale(
            y="shared"
        ).properties(
            height=300
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("まだトレーニングログがありません。週を締めると自動で記録されます。")

    st.markdown("---")

    # ---- ロードマップ（中間ゴール） ----
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

    # ---- MAXテスト機能（テスト日も入力可） ----
    st.subheader("MAXテスト（1RMテスト）")

    st.write("4〜6週間に1度、コンディションの良い日に MAX テストを行うと、正確な1RMが把握できます。")
    st.write("テストを実施した日付と、その日の最大挙上重量（1回挙上できた重量）を入力してください。")

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

    # ==========
    # ダークテーマ + オレンジ系アクセント + トレーニングカード用CSS
    # ==========
    st.markdown("""
    <style>
    /* 全体背景をダーク寄りに */
    [data-testid="stAppViewContainer"] {
        background-color: #020617;
    }
    [data-testid="stSidebar"] {
        background-color: #020617;
    }
    /* 見出しやテキスト色を少し明るめに */
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB;
    }
    .stMarkdown, .stText, .stCaption, .stWrite {
        color: #E5E7EB;
    }

    /* ボタンの色をオレンジ系に */
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

    /* プログレスバーの色もオレンジ */
    .stProgress > div > div {
        background-color: #f97316;
    }

    /* トレーニングメニュー用のカード */
    .workout-card {
        background-color: #111827;
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    openai_client = get_openai_client()

    st.title("💪 ベンチプレス100kgチャレンジアプリ")
    st.caption("Streamlit × ChatGPT 版（Supabase連携はこれから）")

    tabs = st.tabs([
        "初期設定",
        "今週のトレーニング",
        "タンパク質管理",
        "進捗・ロードマップ / テスト",
    ])

    with tabs[0]:
        page_initial_settings()
    with tabs[1]:
        page_training_week(openai_client)
    with tabs[2]:
        page_protein(openai_client)
    with tabs[3]:
        page_progress_and_roadmap()


if __name__ == "__main__":
    main()

#Supabaseクライアントの初期化
from supabase import create_client, Client

def get_supabase_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

#データ保存処理の追加（例：トレーニングログ）
def save_training_log_to_supabase(log: Dict[str, Any]):
    supabase = get_supabase_client()
    data, count = supabase.table("training_logs").insert(log).execute()
    return data

#データの読み込み（例：プロフィール）
def load_profile_from_supabase(user_id: str) -> Optional[Dict[str, Any]]:
    supabase = get_supabase_client()
    res = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
    if res.data:
        return res.data[0]
    return None

