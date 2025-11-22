import streamlit as st # フロントエンドを扱うstreamlitの機能をインポート
from openai import OpenAI # openAIのchatGPTのAIを活用するための機能をインポート

# OpenAI APIの設定
# Streamlit Cloud用とローカル用の両方に対応
try:
    # Streamlit Cloudの場合
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    # ローカル環境の場合
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

# openAIの機能をclientに代入
client = OpenAI(api_key=api_key)


# chatGPTが可能な文章のテイストの設定一覧を作成
content_kind_of =[
    "中立的で客観的な文章",
    "分かりやすい、簡潔な文章",
    "親しみやすいトーンの文章",
    "専門用語をできるだけ使わない、一般読者向けの文章",
    "言葉の使い方にこだわり、正確な表現を心がけた文章",
    "ユーモアを交えた文章",
    "シンプルかつわかりやすい文法を使った文章",
    "面白く、興味深い内容を伝える文章",
    "具体的でイメージしやすい表現を使った文章",
    "人間味のある、感情や思いを表現する文章",
    "引用や参考文献を適切に挿入した、信頼性の高い文章",
    "読み手の興味を引きつけるタイトルやサブタイトルを使った文章",
    "統計データや図表を用いたわかりやすい文章",
    "独自の見解や考え方を示した、論理的な文章",
    "問題提起から解決策までを網羅した、解説的な文章",
    "ニュース性の高い、旬なトピックを取り上げた文章",
    "エンターテイメント性のある、軽快な文章",
    "読者の関心に合わせた、専門的な内容を深く掘り下げた文章",
    "人物紹介やインタビューを取り入れた、読み物的な文章",
]

# chatGPTにリクエストするためのメソッドを設定。引数には書いてほしい内容と文章のテイストと最大文字数を指定
def run_gpt(content_text_to_gpt,content_kind_of_to_gpt,content_maxStr_to_gpt):
    # リクエスト内容を決める
    request_to_gpt = content_text_to_gpt + " また、これを記事として読めるように、記事のタイトル、目次、内容の順番で出力してください。内容は"+ content_maxStr_to_gpt + "文字以内で出力してください。" + "また、文章は" + content_kind_of_to_gpt + "にしてください。"
    
    # 決めた内容を元にopenai.chat.completions.createでchatGPTにリクエスト。オプションとしてmodelにAIモデル、messagesに内容を指定
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": request_to_gpt },
        ],
    )

    # 返って来たレスポンスの内容はresponse.choices[0].message.content.strip()に格納されているので、これをoutput_contentに代入
    output_content = response.choices[0].message.content.strip()
    return output_content # 返って来たレスポンスの内容を返す

st.title('GPTに記事書かせるアプリ')# タイトル
output_content = st.empty() # chatGPTから出力された文字を代入するための箱を用意

select_box = ["シンプルモード", "箇条書きモード"] # モード選択のための選択肢を設定
radio_select = st.sidebar.radio("入力モード", (select_box)) # サイドバーにモード選択のための選択肢を表示

# select_box = ["シンプルモード", "箇条書きモード"]の０番目を選んだ場合は、シンプルモードの設定が表示される
if (radio_select == select_box[0]):
    content_text_to_gpt = st.sidebar.text_input("書かせたい内容を入力してください！") # 書かせたい内容を入力するための欄を表示
else:
    content_text_to_gpt = "" # 箇条書きから生成した書かせたい内容を代入するための箱を用意
    content_text_to_gpt_array = [] # 空欄を排除した箇条書きの内容を整理するための配列を用意する

    content_text_to_gpt_list = [] # 箇条書きで入力された内容を格納するための配列を用意
    content_text_to_gpt_list.append(st.sidebar.text_input("書かせたい内容を箇条書きで入力してください",placeholder="箇条書き１つ目"))
    content_text_to_gpt_list.append(st.sidebar.text_input("項目2つ目"))
    content_text_to_gpt_list.append(st.sidebar.text_input("項目3つ目"))
    content_text_to_gpt_list.append(st.sidebar.text_input("項目4つ目"))
    content_text_to_gpt_list.append(st.sidebar.text_input("項目5つ目"))

    # 入力された配列から空欄を排除し、content_text_to_gpt_arrayに代入
    for c in content_text_to_gpt_list:
        if c != "":
            content_text_to_gpt_array.append(c)


    # 整理した結果、１つでも内容がある場合、書かせたい内容に変換する
    if content_text_to_gpt_array != []:
        content_text_to_gpt = "記事にしてほしい内容を箇条書きにすると、" + "、".join(content_text_to_gpt_array) + " です。"


            
# 書かせたい内容のテイストを選択肢として表示する
content_kind_of_to_gpt = st.sidebar.selectbox("文章の種類",options=content_kind_of)

# chatGPTに出力させる文字数をスライドバーで表示する
content_maxStr_to_gpt = str(st.sidebar.slider('記事の最大文字数', 100,1000,3000))

# エラーが起きたときに表示するための箱をサイドバーに用意する
warning_text = st.sidebar.empty()

# ボタンを押したら、実行
if st.sidebar.button('記事を書かせる'):

    # 無駄にリクエストしないように書かせたい内容に中身があるか確認し、あれば実行
    if (content_text_to_gpt != ""):
        output_content.write("GPT生成中") # 状況案内を表示
        warning_text.write("") # 正常に実行されているので、エラーを書き込む箱を空欄書き込みでリセット処理

        # chatGPTにリクエストするためのメソッドを設定。引数には書いてほしい内容と文章のテイストと最大文字数を指定してメソッド実行し、output_content_textに代入
        output_content_text = run_gpt(content_text_to_gpt,content_kind_of_to_gpt,content_maxStr_to_gpt)

        # 代入された文字を表示
        output_content.write(output_content_text) 

        # 代入された文字をダウンロードするボタンを設置。オプションは内容をdataに指定、ファイル名をfile_nameに指定、ファイルタイプをmimeに指定
        st.download_button(label='記事内容 Download', 
                   data=output_content_text, 
                   file_name='out_put.txt',
                   mime='text/plain',
                   )
    else:
        warning_text.write("書かせたい内容が入力されていません") # 書かせたい内容がないので、エラーとして、用意したエラーの場所に書き込み


    