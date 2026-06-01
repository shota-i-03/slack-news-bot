#!/usr/bin/env python3
"""
AI/MI RSS News Fetcher → Groq (Llama 3.3, 無料) 要約 → Slack Webhook投稿

#ai-news    : AI全般の最新ニュース・ツール・トレンドを魅力的に
#ai-research: MI関連の研究論文をわかりやすく構造化して解説
"""

import argparse
import os
import feedparser
import requests
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# ─── 設定 ───────────────────────────────────────────────────────────────────

HOURS_LOOKBACK = 24
MAX_ARTICLES_PER_FEED = 8
MAX_ARTICLES_FOR_LLM = 15
GROQ_MODEL = "llama-3.3-70b-versatile"

# #ai-news: 実用的なニュース・ツール・業界動向中心（論文は少なめ）
AI_NEWS_FEEDS: list[tuple[str, str]] = [
    ("Hacker News (AI)",    "https://hnrss.org/newest?q=AI+LLM+machine+learning&points=50"),
    ("TechCrunch AI",       "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("VentureBeat AI",      "https://venturebeat.com/ai/feed/"),
    ("The Verge AI",        "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml"),
    ("Hugging Face Blog",   "https://huggingface.co/blog/feed.xml"),
    ("DeepMind Blog",       "https://deepmind.google/blog/rss.xml"),
]

# #stock-news: 個人投資家向け株・市場ニュース
STOCK_NEWS_FEEDS: list[tuple[str, str]] = [
    ("NHK経済",           "https://www.nhk.or.jp/rss/news/cat5.xml"),
    ("Reuters Japan",     "https://jp.reuters.com/rssFeed/businessNews"),
    ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
    ("Investing.com JP",  "https://jp.investing.com/rss/news.rss"),
    ("東洋経済オンライン", "https://toyokeizai.net/list/feed/rss"),
]

# #ai-research: Materials Informatics / 材料科学 × AI の研究論文中心
AI_RESEARCH_FEEDS: list[tuple[str, str]] = [
    ("arXiv cond-mat.mtrl-sci", "http://arxiv.org/rss/cond-mat.mtrl-sci"),
    ("arXiv cs.LG (MI応用)",    "http://arxiv.org/rss/cs.LG"),
    ("arXiv cs.AI",             "http://arxiv.org/rss/cs.AI"),
    ("npj Comp. Materials",     "https://www.nature.com/npjcompumats.rss"),
    ("NIMS NEWS",               "https://www.nims.go.jp/news/rss.xml"),
]

CHANNELS: list[tuple[str, list, str, str]] = [
    ("ai_news",     AI_NEWS_FEEDS,     "SLACK_WEBHOOK_AI_NEWS",     "今日のAIニュース"),
    ("ai_research", AI_RESEARCH_FEEDS, "SLACK_WEBHOOK_AI_RESEARCH", "今日のMI研究ピックアップ"),
    ("stock_news",  STOCK_NEWS_FEEDS,  "SLACK_WEBHOOK_STOCK_NEWS",  "今朝の株・市場ニュース"),
]

# ─── プロンプト ───────────────────────────────────────────────────────────────

NEWS_PROMPT = """\
以下は本日のAI関連ニュース・リリース一覧です。
エンジニアや研究者にとって重要なトップ5を選び、以下の形式で日本語にまとめてください。

ルール:
- 難しい専門用語はできるだけ避ける
- 「なぜこれが重要か」「何が変わるか」を必ず一言入れる
- 絵文字は使わない

形式（5件）:
**[タイトル]**
→ 内容を2〜3文で。重要な理由・インパクトも一言添える。
URL

---

記事一覧:
{articles_text}
"""

RESEARCH_PROMPT = """\
以下はMI（マテリアルズ・インフォマティクス）・材料科学×AI の最新論文・研究記事一覧です。
最も注目すべき論文を2〜3本選び、それぞれを以下の形式で「研究に不慣れな人でも理解できるよう」日本語で解説してください。

ルール:
- 専門用語は出てきたら必ずひと言で補足する
- 高校生でもなんとなくわかるくらいの平易さを目指す
- 各項目は1〜3文に収める
- 絵文字は使わない

形式（論文ごとに繰り返す）:
━━━━━━━━━━━━━━━━━━━━━━━━━━
**[論文タイトル（日本語訳）]**
URL

どんなもの？
→

先行研究と比べてどこがすごい？
→

技術・手法のキモはどこ？
→

どうやって有効だと検証した？
→

議論・課題はある？
→
━━━━━━━━━━━━━━━━━━━━━━━━━━

---

論文一覧:
{articles_text}
"""

STOCK_PROMPT = """\
以下は本日の株式・金融市場に関するニュース一覧です。
個人投資家向けに、3カテゴリに分けてそれぞれ重要度順に4件ずつ日本語でまとめてください。

ルール:
- 株価・市場・企業業績・経済指標など投資判断に直結する情報を優先する
- 各ニュースは「一言で何が起きたか」＋「投資家への影響」の2文以内に収める
- 絵文字は使わない
- 記事が少ないカテゴリは書ける分だけ書く

以下の形式を厳守してください:

━━━ 米国関連 ━━━

1.  タイトル
    内容（2文以内）
    URL

2. ...

━━━ 日本関連 ━━━

1.  タイトル
    内容（2文以内）
    URL

2. ...

━━━ その他・主要指数 ━━━
（米国・日本以外の市場／日経平均・S&P500・ナスダック・為替・コモディティ）

1.  タイトル
    内容（2文以内）
    URL

2. ...

---

記事一覧:
{articles_text}
"""

# ─── RSS取得 ─────────────────────────────────────────────────────────────────

def fetch_recent_articles(feeds: list[tuple[str, str]]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_LOOKBACK)
    articles = []

    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                if published and published < cutoff:
                    continue

                articles.append({
                    "source": source_name,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:600],
                    "link": entry.get("link", ""),
                })
        except Exception as e:
            print(f"[WARNING] {source_name} 取得失敗: {e}")

    return articles


# ─── LLM要約（Groq / 無料）────────────────────────────────────────────────────

def summarize(articles: list[dict], category: str) -> str:
    if not articles:
        return "本日は新着記事がありませんでした。"

    articles_text = "\n\n".join([
        f"[{a['source']}] {a['title']}\n概要: {a['summary']}\nURL: {a['link']}"
        for a in articles[:MAX_ARTICLES_FOR_LLM]
    ])

    if category == "ai_news":
        prompt_template = NEWS_PROMPT
    elif category == "stock_news":
        prompt_template = STOCK_PROMPT
    else:
        prompt_template = RESEARCH_PROMPT
    prompt = prompt_template.format(articles_text=articles_text)

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    print(f"  トークン: 入力={usage.prompt_tokens}, 出力={usage.completion_tokens}, 合計={usage.total_tokens}")
    return response.choices[0].message.content


# ─── Slack投稿 ───────────────────────────────────────────────────────────────

def post_to_slack(webhook_url: str, title: str, body: str) -> None:
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst).strftime("%Y-%m-%d %H:%M JST")
    payload = {"text": f"*{title}*  _{now}_\n\n{body}"}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["ai_news", "ai_research", "stock_news"], help="実行するチャンネルを指定（省略時は全て）")
    args = parser.parse_args()

    targets = [ch for ch in CHANNELS if args.channel is None or ch[0] == args.channel]

    for category, feeds, webhook_env, label in targets:
        webhook = os.environ.get(webhook_env)
        if not webhook:
            print(f"[SKIP] {webhook_env} が未設定のため {label} をスキップします")
            continue

        print(f"\n▶ {label} を処理中...")
        articles = fetch_recent_articles(feeds)
        print(f"  取得記事数: {len(articles)}")

        summary = summarize(articles, category)
        post_to_slack(webhook, label, summary)
        print(f"  Slack投稿完了")


if __name__ == "__main__":
    main()
