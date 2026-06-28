import os, re, json, argparse
import configparser
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv

try:
    from notion_client import Client
except Exception:
    Client = None

# --- Language utilities ---
JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")  # Hiragana/Katakana/Kanji
CODE_FENCE_RE = re.compile(r"```")
INLINE_CODE_RE = re.compile(r"(?<!`)`(?!`)")
LINK_RE = re.compile(r"\[[^\]]*\]\([^\)]+\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
HEADING_RE = re.compile(r"^#+\s", re.MULTILINE)

try:
    from langdetect import detect
except Exception:
    def detect(_):
        return "unknown"


def get_text_from_property(page: Dict[str, Any], prop_name: str) -> str:
    props = page.get("properties", {})
    if prop_name not in props:
        return ""
    p = props[prop_name]
    t = p.get("type")
    if t == "title":
        return "".join([x.get("plain_text", "") for x in p.get("title", [])]).strip()
    if t == "rich_text":
        return "".join([x.get("plain_text", "") for x in p.get("rich_text", [])]).strip()
    # Support for other text-like types if needed
    return ""


def md_counts(text: str) -> Dict[str, int]:
    return {
        "code_fence": len(CODE_FENCE_RE.findall(text)),
        "inline_code": len(INLINE_CODE_RE.findall(text)),
        "links": len(LINK_RE.findall(text)),
        "images": len(IMG_RE.findall(text)),
        "headings": len(HEADING_RE.findall(text)),
    }


def contains_japanese(text: str) -> bool:
    return bool(JP_RE.search(text or ""))


def length_ratio_issue(src: str, tgt: str) -> bool:
    s = max(len((src or "").strip()), 1)
    t = max(len((tgt or "").strip()), 1)
    r = t / s
    return (r < 0.35) or (r > 2.5)


def lang_is_expected(text: str, expected: str) -> bool:
    try:
        d = detect(text) if text else "unknown"
    except Exception:
        d = "unknown"
    if expected == "en":
        return d == "en"
    if expected == "zh-cn":
        return d in ("zh-cn", "zh")
    if expected == "zh-tw":
        return d in ("zh-tw", "zh")
    return True


def compare_md_structure(jp: str, tgt: str) -> List[str]:
    issues = []
    c1, c2 = md_counts(jp or ""), md_counts(tgt or "")
    for k in c1:
        if c1[k] != c2[k]:
            issues.append(f"Markdown構造差異: {k} JP={c1[k]} / TGT={c2[k]}")
    return issues


def check_translation(jp: str, tgt: str, expected_lang: str) -> List[str]:
    issues = []
    if not tgt:
        issues.append("未翻訳（空文字）")
        return issues
    if not lang_is_expected(tgt, expected_lang):
        issues.append("言語判定不一致")
    if contains_japanese(tgt) and expected_lang in ("en", "zh-cn", "zh-tw"):
        issues.append("日本語の残存が疑われます")
    if length_ratio_issue(jp, tgt):
        issues.append("長さ比の異常（過小/過大）")
    issues += compare_md_structure(jp, tgt)
    # 画像URL/リンクURLはそのまま維持されるべき
    # 異常検知: JPとTGTでURL数が極端に差異
    jp_links, tgt_links = len(LINK_RE.findall(jp or "")), len(LINK_RE.findall(tgt or ""))
    jp_imgs, tgt_imgs = len(IMG_RE.findall(jp or "")), len(IMG_RE.findall(tgt or ""))
    if abs(jp_links - tgt_links) > 0:
        issues.append("リンク数の不整合")
    if abs(jp_imgs - tgt_imgs) > 0:
        issues.append("画像数の不整合")
    return issues

# --- LLM-based consistency check (optional) ---

def llm_consistency_check_openai(jp: str, tgt: str, expected_lang: str, model: str) -> Dict[str, Any]:
    """OpenAIを使った原文/訳文の整合性チェック。JSONで判定を返す。"""
    try:
        from openai import OpenAI
    except Exception:
        return {"status": "error", "error": "openaiパッケージ未導入"}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "skip", "reason": "OPENAI_API_KEY未設定"}

    client = OpenAI()
    # 入力を短縮し、過度なトークンを避ける
    jp_s = (jp or "").strip()
    tgt_s = (tgt or "").strip()
    jp_s = jp_s[:4000]
    tgt_s = tgt_s[:4000]

    system = (
        "あなたは翻訳監査員です。Markdown構造と意味の整合性を厳密に評価します。"
        "返答は必ずJSONのみで、説明文は一切含めないでください。"
    )
    user = (
        f"原文(JP):\n{jp_s}\n\n訳文({expected_lang}):\n{tgt_s}\n\n"
        "評価基準:\n"
        "- 意味の一致・用語の一貫性\n"
        "- 重要情報の欠落・追加の有無\n"
        "- Markdown構造（見出し/リンク/画像/コード）の維持\n"
        "出力形式(JSONのみ):\n"
        "{\n  \"verdict\": \"ok\" または \"issue\",\n"
        "  \"reasons\": [\"理由を箇条書き\"],\n"
        "  \"severity\": \"low\"|\"medium\"|\"high\"\n}")
        
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        # 念のためJSON抽出を試みる
        json_str = content
        try:
            data = json.loads(json_str)
        except Exception:
            # 最小限の抽出(先頭{から末尾}まで)
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    return {"status": "error", "error": "LLM応答がJSON形式ではありません"}
            else:
                return {"status": "error", "error": "LLM応答がJSON形式ではありません"}
        verdict = (data.get("verdict") or "").lower()
        if verdict == "ok":
            return {"status": "ok"}
        elif verdict == "issue":
            return {"status": "issue", "reasons": data.get("reasons", []), "severity": data.get("severity", "medium")}
        else:
            return {"status": "error", "error": "verdict未定義"}
    except Exception as e:
        return {"status": "error", "error": f"LLM呼び出し失敗: {type(e).__name__}: {str(e)}"}


def llm_consistency_check(jp: str, tgt: str, expected_lang: str, provider: str, model: str) -> Dict[str, Any]:
    provider = (provider or "").lower()
    if provider == "openai":
        return llm_consistency_check_openai(jp, tgt, expected_lang, model)
    return {"status": "skip", "reason": "未対応プロバイダ"}


def summarize_llm_issue(res: Dict[str, Any]) -> str:
    reasons = res.get("reasons") or []
    if isinstance(reasons, list) and reasons:
        reasons_text = "; ".join([str(r) for r in reasons[:3]])
    else:
        reasons_text = str(reasons) if reasons else ""
    sev = res.get("severity", "medium")
    return f"LLM整合性: 不一致(severity={sev})" + (f" - {reasons_text}" if reasons_text else "")


def append_lang_issue(acc: List[Dict[str, Any]], lang: str, issue_text: str) -> None:
    for item in acc:
        if item.get("lang") == lang:
            item.setdefault("issues", []).append(issue_text)
            return
    acc.append({"lang": lang, "issues": [issue_text]})


def query_notion_pages(token: str, database_id: str, since_days: int) -> List[Dict[str, Any]]:
    if Client is None:
        raise RuntimeError("notion-clientが未インストールです。'pip install notion-client' を実行してください。")
    client = Client(auth=token)
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    results = []
    start_cursor = None
    while True:
        payload = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor
        # フィルタ: 最近編集されたページ
        payload["filter"] = {"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": since_date}}
        resp = client.databases.query(database_id=database_id, **payload)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        start_cursor = resp.get("next_cursor")
    return results


def read_notion_config(config_path=None):
    path = config_path or os.path.join(os.path.dirname(__file__), "config.ini")
    database_id = None
    token = None
    try:
        cfg = configparser.ConfigParser()
        if os.path.exists(path):
            cfg.read(path, encoding="utf-8")
            if cfg.has_section("notion"):
                token = cfg.get("notion", "notion_token", fallback=None)
                database_id = cfg.get("notion", "notion_database_id", fallback=None)
    except Exception:
        pass
    return database_id, token


def safe_slug(name: str, default: str) -> str:
    # 記号を落とし、スペースをアンダースコアに変換し、長さ制限
    slug = re.sub(r"[^0-9a-zA-Z\u3040-\u30FF\u4E00-\u9FFF\-_. ]+", "", (name or "")).strip()
    if not slug:
        return default
    slug = slug.replace(" ", "_")
    return slug[:80]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_page_url(page: Dict[str, Any]) -> str:
    url = page.get("url")
    if url:
        return url
    pid = (page.get("id", "") or "").replace("-", "")
    return f"https://www.notion.so/{pid}" if pid else ""


def fetch_page_blocks(token: str, page_id: str) -> List[Dict[str, Any]]:
    if Client is None or not token or not page_id:
        return []
    try:
        client = Client(auth=token)
        blocks: List[Dict[str, Any]] = []
        start_cursor = None
        while True:
            if start_cursor:
                resp = client.blocks.children.list(block_id=page_id, start_cursor=start_cursor)
            else:
                resp = client.blocks.children.list(block_id=page_id)
            blocks.extend(resp.get("results", []))
            if resp.get("has_more"):
                start_cursor = resp.get("next_cursor")
            else:
                break
        return blocks
    except Exception:
        return []


def _rich_text_to_plain(rich_text: List[Dict[str, Any]]) -> str:
    return "".join([span.get("plain_text", "") for span in (rich_text or [])])


def render_blocks_markdown(blocks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for b in blocks:
        t = b.get("type")
        data = b.get(t, {})
        if t == "heading_1":
            lines.append("# " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "heading_2":
            lines.append("## " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "heading_3":
            lines.append("### " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "paragraph":
            lines.append(_rich_text_to_plain(data.get("rich_text", [])))
        elif t == "bulleted_list_item":
            lines.append("- " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "numbered_list_item":
            lines.append("1. " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "to_do":
            checked = data.get("checked", False)
            lines.append(f"- [{'x' if checked else ' '}] " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "quote":
            lines.append("> " + _rich_text_to_plain(data.get("rich_text", [])))
        elif t == "code":
            lang = data.get("language", "")
            txt = _rich_text_to_plain(data.get("rich_text", []))
            lines.append(f"```{lang}\n{txt}\n```")
        else:
            # 未対応のブロックは種類のみ表示
            lines.append(f"[{t}]")
    return "\n\n".join([ln for ln in lines if ln])


def export_results(report: List[Dict[str, Any]], output_dir: str, export_format: str, export_mode: str, write_blocks: bool, token: str) -> None:
    ensure_dir(output_dir)
    ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if export_mode == "aggregate":
        if export_format == "json":
            path = os.path.join(output_dir, f"issues_{ts_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"count": len(report), "items": report}, f, ensure_ascii=False, indent=2)
        else:
            path = os.path.join(output_dir, f"issues_{ts_name}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Notion翻訳問題一覧 ({len(report)}件)\n\n")
                for item in report:
                    f.write(f"## {item['title']}\n\n")
                    f.write(f"- ページURL: {item.get('url','')}\n")
                    f.write(f"- 最終更新: {item.get('last_edited_time','')}\n\n")
                    f.write("### 原文(JP)\n\n")
                    f.write((item.get('jp') or '') + "\n\n")
                    if item.get("en") is not None:
                        f.write("### 英語(EN)\n\n")
                        f.write((item.get('en') or '') + "\n\n")
                    if item.get("zhcn") is not None:
                        f.write("### 中国語(簡体)\n\n")
                        f.write((item.get('zhcn') or '') + "\n\n")
                    if item.get("zhtw") is not None:
                        f.write("### 中国語(繁体)\n\n")
                        f.write((item.get('zhtw') or '') + "\n\n")
                    f.write("### 検出問題\n\n")
                    for p in item['problems']:
                        f.write(f"- {p['lang']}: " + "; ".join(p['issues']) + "\n")
                    if write_blocks:
                        blocks = fetch_page_blocks(token, item.get('page_id'))
                        if blocks:
                            f.write("\n### ページ本文スナップショット\n\n")
                            f.write(render_blocks_markdown(blocks) + "\n")
        return

    # per_page
    for item in report:
        slug = safe_slug(item['title'], (item.get('page_id','') or '').replace('-', '') or 'page')
        if export_format == "json":
            path = os.path.join(output_dir, f"{slug}.json")
            data = dict(item)
            if write_blocks:
                data["blocks_markdown"] = render_blocks_markdown(fetch_page_blocks(token, item.get('page_id')))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            path = os.path.join(output_dir, f"{slug}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {item['title']}\n\n")
                f.write(f"- ページURL: {item.get('url','')}\n")
                f.write(f"- 最終更新: {item.get('last_edited_time','')}\n\n")
                f.write("## 原文(JP)\n\n")
                f.write((item.get('jp') or '') + "\n\n")
                if item.get("en") is not None:
                    f.write("## 英語(EN)\n\n")
                    f.write((item.get('en') or '') + "\n\n")
                if item.get("zhcn") is not None:
                    f.write("## 中国語(簡体)\n\n")
                    f.write((item.get('zhcn') or '') + "\n\n")
                if item.get("zhtw") is not None:
                    f.write("## 中国語(繁体)\n\n")
                    f.write((item.get('zhtw') or '') + "\n\n")
                f.write("## 検出問題\n\n")
                for p in item['problems']:
                    f.write(f"- {p['lang']}: " + "; ".join(p['issues']) + "\n")
                if write_blocks:
                    blocks_md = render_blocks_markdown(fetch_page_blocks(token, item.get('page_id')))
                    if blocks_md:
                        f.write("\n## ページ本文スナップショット\n\n")
                        f.write(blocks_md + "\n")


def main():
    load_dotenv()
    # config.iniから既定値を取得
    db_id_def, token_def = read_notion_config()
    parser = argparse.ArgumentParser(description="Notion翻訳チェック（問題ページのみダウンロード・出力）")
    parser.add_argument("--database-id", default=db_id_def, help="NotionデータベースID")
    parser.add_argument("--token", default=token_def, help="Notion統合トークン")
    parser.add_argument("--since-days", type=int, default=int(os.getenv("NOTION_SINCE_DAYS", "7")), help="何日前までを対象にするか")
    parser.add_argument("--prop-ja", default=os.getenv("NOTION_FIELD_JA", "JP"))
    parser.add_argument("--prop-en", default=os.getenv("NOTION_FIELD_EN", "EN"))
    parser.add_argument("--prop-zhcn", default=os.getenv("NOTION_FIELD_ZH_CN", "ZH_CN"))
    parser.add_argument("--prop-zhtw", default=os.getenv("NOTION_FIELD_ZH_TW", "ZH_TW"))
    parser.add_argument("--format", choices=["markdown","json"], default=os.getenv("OUTPUT_FORMAT","markdown"))
    # 出力関連オプション
    parser.add_argument("--output-dir", default=os.getenv("ISSUE_OUTPUT_DIR","notion_issue_exports"), help="問題ページの出力ディレクトリ")
    parser.add_argument("--export-format", choices=["markdown","json"], default=os.getenv("EXPORT_FORMAT","markdown"), help="ファイル出力形式")
    parser.add_argument("--export-mode", choices=["per_page","aggregate"], default=os.getenv("EXPORT_MODE","per_page"), help="出力モード（ページ毎/集約）")
    parser.add_argument("--write-files", dest="write_files", action="store_true", default=True, help="問題ページをファイル出力（既定ON）")
    parser.add_argument("--no-write-files", dest="write_files", action="store_false", help="ファイル出力を無効化")
    parser.add_argument("--download-blocks", dest="download_blocks", action="store_true", default=True, help="問題ページ本文も取得して出力（既定ON）")
    parser.add_argument("--no-download-blocks", dest="download_blocks", action="store_false", help="本文取得を無効化")
    # 追加: LLM整合性チェックオプション
    parser.add_argument("--llm-check", dest="llm_check", action="store_true", default=True, help="LLMによる整合性追加判定を有効化（既定ON）")
    parser.add_argument("--no-llm-check", dest="llm_check", action="store_false", help="LLM整合性チェックを無効化")
    parser.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER","openai"), help="LLMプロバイダ（現在openai対応）")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL","gpt-3.5-turbo"), help="使用モデル名（プロバイダに依存）")
    args = parser.parse_args()

    if not args.database_id or not args.token:
        print("ERROR: config.iniの[notion]にnotion_database_id/notion_tokenを設定するか、引数で指定してください。")
        return 1

    pages = query_notion_pages(args.token, args.database_id, args.since_days)
    report = []
    for page in pages:
        title = get_text_from_property(page, "Name") or get_text_from_property(page, args.prop_ja) or page.get("id","")
        jp = get_text_from_property(page, args.prop_ja)
        en = get_text_from_property(page, args.prop_en)
        zhcn = get_text_from_property(page, args.prop_zhcn)
        zhtw = get_text_from_property(page, args.prop_zhtw)

        page_issues = []
        # 英語
        if en is not None:
            iss = check_translation(jp, en, "en")
            if iss:
                page_issues.append({"lang":"EN","issues":iss})
            if args.llm_check and en:
                res = llm_consistency_check(jp, en, "en", args.llm_provider, args.llm_model)
                if res.get("status") == "issue":
                    append_lang_issue(page_issues, "EN", summarize_llm_issue(res))
        # 中国語(簡体)
        if zhcn is not None:
            iss = check_translation(jp, zhcn, "zh-cn")
            if iss:
                page_issues.append({"lang":"ZH_CN","issues":iss})
            if args.llm_check and zhcn:
                res = llm_consistency_check(jp, zhcn, "zh-cn", args.llm_provider, args.llm_model)
                if res.get("status") == "issue":
                    append_lang_issue(page_issues, "ZH_CN", summarize_llm_issue(res))
        # 中国語(繁体)
        if zhtw is not None:
            iss = check_translation(jp, zhtw, "zh-tw")
            if iss:
                page_issues.append({"lang":"ZH_TW","issues":iss})
            if args.llm_check and zhtw:
                res = llm_consistency_check(jp, zhtw, "zh-tw", args.llm_provider, args.llm_model)
                if res.get("status") == "issue":
                    append_lang_issue(page_issues, "ZH_TW", summarize_llm_issue(res))

        if page_issues:
            report.append({
                "page_id": page.get("id",""),
                "url": get_page_url(page),
                "title": title,
                "last_edited_time": page.get("last_edited_time",""),
                "jp": jp,
                "en": en,
                "zhcn": zhcn,
                "zhtw": zhtw,
                "problems": page_issues,
            })

    # ファイル出力（問題ページのみ）
    if args.write_files and report:
        export_results(report, args.output_dir, args.export_format, args.export_mode, args.download_blocks, args.token)

    # コンソール出力
    if args.format == "json":
        print(json.dumps({"count": len(report), "items": report}, ensure_ascii=False, indent=2))
    else:
        print(f"対象ページ: {len(pages)}件 / 問題検出: {len(report)}件")
        if args.write_files and report:
            print(f"問題ページを書き出しました: {args.output_dir}（{args.export_mode} / {args.export_format}）")
        for item in report:
            print(f"- {item['title']} ({item['page_id']}) [last_edited: {item['last_edited_time']}]")
            for p in item["problems"]:
                print(f"  - {p['lang']}: " + "; ".join(p['issues']))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())