"""Query Server for searching relevant diaries using embeddings."""

import os
import re
import json
import tomllib
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Optional, Any
from datetime import datetime


# Constants
DATE_DIR_PATTERN = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$"
DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"
SIMPLE_DATE_FORMAT = "%Y-%m-%d"
HALF_LIFE_DAYS = 30
LN_2 = 0.693
SECONDS_PER_DAY = 86400
DEFAULT_NUM_RESULTS = 5
DEFAULT_RECENCY_WEIGHT = 0.5
MAX_NUM_DIARIES = 15
EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-pro"
DEFAULT_TITLE = "无标题"


class QueryServer:
    """Server for querying diaries using embedding similarity search"""
    
    def __init__(self):
        print("Initializing QueryServer...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        config_path = os.path.join(project_root, "config", "config.toml")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        
        genai.configure()
        self.project_root = project_root
        
        print("Loading all diaries...")
        self._load_all_diaries()
        print(f"Loaded {len(self.diaries)} diaries into memory")
    
    def _load_all_diaries(self):
        """Load all diary data and embeddings into memory."""
        dir_names = [
            name for name in os.listdir(self.config["diary_dir"]) 
            if re.match(DATE_DIR_PATTERN, name)
        ]
        dir_names.sort()
        
        self.diaries = []
        embeddings = []
        diary_dates = []
        
        for dir_name in dir_names:
            try:
                vec_path = os.path.join(self.config["diary_dir"], dir_name, self.config["embedding_name"])
                if not os.path.exists(vec_path):
                    continue
                
                vec = np.load(vec_path)
                
                diary_path = os.path.join(self.config["diary_dir"], dir_name, self.config["diary_name"])
                title_path = os.path.join(self.config["diary_dir"], dir_name, self.config["title_name"])
                
                content = self._read_file(diary_path)
                title = self._read_file(title_path).strip() or DEFAULT_TITLE
                
                dt_obj = datetime.strptime(dir_name, DATE_FORMAT)
                
                self.diaries.append({
                    "date": dir_name,
                    "datetime": dt_obj,
                    "title": title,
                    "content": content
                })
                embeddings.append(vec)
                diary_dates.append(dt_obj.timestamp())
                
            except Exception as e:
                print(f"Error loading {dir_name}: {e}")
                continue
        
        self.embedding_matrix = np.array(embeddings, dtype=np.float32)
        self.embedding_norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        self.diary_timestamps = np.array(diary_dates, dtype=np.float32)
    
    def _read_file(self, path: str) -> str:
        """Read file content safely."""
        if not os.path.exists(path):
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query string."""
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        return np.array(response['embedding'], dtype=np.float32)
    
    def query(
        self, 
        query: str, 
        num: int = DEFAULT_NUM_RESULTS, 
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant diaries using embedding similarity with time decay.
        
        Args:
            query: Search query string
            num: Number of results to return
            recency_weight: Time decay weight (0.0 = semantic only, 1.0 = high time bias)
            start_date: Filter diaries after this date (format: 'YYYY-MM-DD', inclusive)
            end_date: Filter diaries before this date (format: 'YYYY-MM-DD', inclusive)
        
        Returns:
            List of dictionaries with keys: date, title, content, score, semantic, freshness
            Sorted by relevance score (highest to lowest)
        """
        query_vec = self._get_query_embedding(query)
        query_norm = np.linalg.norm(query_vec)
        
        semantic_scores = (self.embedding_matrix @ query_vec) / (self.embedding_norms.flatten() * query_norm)
        
        mask = np.ones(len(self.diaries), dtype=bool)
        
        if start_date:
            try:
                start_ts = datetime.strptime(start_date, SIMPLE_DATE_FORMAT).replace(
                    hour=0, minute=0, second=0
                ).timestamp()
                mask = mask & (self.diary_timestamps >= start_ts)
                print(f"[Filter] Applied start_date: {start_date} (matched {np.sum(mask)} diaries)")
            except ValueError:
                print(f"[Warning] Invalid start_date format: {start_date}, ignoring filter")
        
        if end_date:
            try:
                end_ts = datetime.strptime(end_date, SIMPLE_DATE_FORMAT).replace(
                    hour=23, minute=59, second=59
                ).timestamp()
                mask = mask & (self.diary_timestamps <= end_ts)
                print(f"[Filter] Applied end_date: {end_date} (matched {np.sum(mask)} diaries)")
            except ValueError:
                print(f"[Warning] Invalid end_date format: {end_date}, ignoring filter")
        
        if not np.any(mask):
            print(f"[Warning] No diaries found in date range [{start_date} ~ {end_date}]")
            return []
        
        valid_indices = np.where(mask)[0]
        
        current_ts = datetime.now().timestamp()
        time_diffs = current_ts - self.diary_timestamps[valid_indices]
        days_ago = time_diffs / SECONDS_PER_DAY
        
        freshness_scores = np.exp(-LN_2 * days_ago / HALF_LIFE_DAYS)
        
        valid_semantic_scores = semantic_scores[valid_indices]
        final_scores = valid_semantic_scores + (recency_weight * freshness_scores)
        
        top_k = min(num, len(valid_indices))
        top_rel_indices = np.argsort(final_scores)[::-1][:top_k]
        
        top_indices = valid_indices[top_rel_indices]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "date": self.diaries[idx]["date"],
                "title": self.diaries[idx]["title"],
                "content": self.diaries[idx]["content"],
                "score": float(final_scores[top_rel_indices[i]]),
                "semantic": float(semantic_scores[idx]),
                "freshness": float(freshness_scores[top_rel_indices[i]])
            })
        
        return results
    
    def _get_query_refinement_prompt(self, current_date: str) -> str:
        """Get system prompt for query refinement."""
        return (
            f"Current Date: {current_date}\n\n"
            "你是一个智能日记检索专家。你的任务是分析用户问题，提取搜索关键词、时间范围、新鲜度权重和检索数量。"
                    "请输出严格的 JSON 格式（不要使用 Markdown 代码块），包含以下字段：\n"
                    "1. \"refined_query\": (string) 优化后的搜索关键词，包含核心实体、同义词和潜在情感词。\n"
                    "2. \"start_date\": (string or null) 时间范围起始日期，格式 YYYY-MM-DD。如果用户未指定或不需要过滤则为 null。\n"
                    "3. \"end_date\": (string or null) 时间范围结束日期，格式 YYYY-MM-DD。如果用户未指定或不需要过滤则为 null。\n"
                    "4. \"recency_weight\": (float) 时间新鲜度权重，范围 0.0 到 1.0。\n"
                    "5. \"num_diaries\": (int) 需要检索的日记篇数，范围 5 到 15。\n\n"
                    "时间范围提取规则：\n"
                    "- 如果用户明确提到年份/月份（如'2022年'、'去年'、'上个月'），必须计算出具体的 start_date 和 end_date。\n"
                    "- '去年' = 上一个完整年份（例如当前2026年，去年就是2025-01-01到2025-12-31）\n"
                    "- '上个月' = 上一个完整月份（例如当前1月，上个月就是2025-12-01到2025-12-31）\n"
                    "- '上周' = 上一个完整周（周一到周日，例如当前是2026-01-12周日，上周就是2026-01-05到2026-01-11）\n"
                    "- '最近' = 通常指最近7-14天，但不硬性过滤，使用高 recency_weight 即可（start_date和end_date为null）\n"
                    "- 如果用户问'以前'、'曾经'等模糊时间，start_date 和 end_date 均为 null。\n"
                    "- 宁可范围宽一点，也不要漏掉相关日记。\n\n"
                    "新鲜度权重规则（recency_weight）：\n"
                    "- [0.0]: 用户询问特定过去时间点，或已经设置了明确的时间范围过滤。\n"
                    "- [0.1 - 0.2]: 一般性话题，稍微偏向近期。\n"
                    "- [0.5 - 0.8]: 用户询问'最近'但未指定具体日期，需要软性偏向。\n\n"
                    "检索数量判断规则（num_diaries）：\n"
                    "- [5]: 简单事实性问题，有明确答案（如'我昨天吃了什么'、'某个具体日期发生了什么'）。\n"
                    "- [5-7]: 标准问题，需要一定上下文（如'我最近心情怎么样'、'我喜欢什么'、一般性回忆）。\n"
                    "- [8-12]: 复杂分析性问题，需要较多上下文（如'总结我这个月的状态'、'我和某人的关系变化'）。\n"
                    "- [10-15]: 总结性、统计性问题，需要大量数据（如'我今年去过哪些地方'、'分析我的情绪趋势'）。\n\n"
                    "示例 1（硬过滤+中等数量）：\n"
                    "用户：我去年去哪旅游了？\n"
                    "输出：{\"refined_query\": \"旅游 出游 旅行 景点 度假 游玩 飞机 火车\", \"start_date\": \"2025-01-01\", \"end_date\": \"2025-12-31\", \"recency_weight\": 0.0, \"num_diaries\": 10}\n\n"
                    "示例 2（软偏向+标准数量）：\n"
                    "用户：我最近压力好大\n"
                    "输出：{\"refined_query\": \"压力 焦虑 烦躁 心情不好 工作压力 学业压力 最近的状态\", \"start_date\": null, \"end_date\": null, \"recency_weight\": 0.8, \"num_diaries\": 5}\n\n"
                    "示例 3（无过滤+少量）：\n"
                    "用户：我大一的时候喜欢谁？\n"
                    "输出：{\"refined_query\": \"大一 大学一年级 暗恋 喜欢的人 恋爱 感情经历\", \"start_date\": null, \"end_date\": null, \"recency_weight\": 0.0, \"num_diaries\": 5}\n\n"
                    "示例 4（精确日期+标准数量）：\n"
                    "用户：上周我做了什么？\n"
                    "输出：{\"refined_query\": \"日常活动 做的事情 经历 工作 学习 娱乐\", \"start_date\": \"2026-01-05\", \"end_date\": \"2026-01-11\", \"recency_weight\": 0.0, \"num_diaries\": 7}\n\n"
                    "示例 5（简单事实+标准数量）：\n"
                    "用户：我昨天吃了什么？\n"
                    "输出:{\"refined_query\": \"食物 吃饭 餐厅 美食 做饭 外卖\", \"start_date\": \"2026-01-11\", \"end_date\": \"2026-01-11\", \"recency_weight\": 0.0, \"num_diaries\": 5}"
        )
    
    def question(self, user_question: str) -> str:
        """
        Answer a question based on diary content using RAG approach.
        
        Args:
            user_question: The question from user
        
        Returns:
            The answer from GPT based on retrieved diary context
        """
        current_date_str = datetime.now().strftime(SIMPLE_DATE_FORMAT)
        
        print("\n[Step 1] Analyzing intent with Gemini...")
        
        sys_instruction = self._get_query_refinement_prompt(current_date_str)
        model = genai.GenerativeModel(
             model_name=CHAT_MODEL,
             system_instruction=sys_instruction,
             generation_config={"response_mime_type": "application/json"}
        )
        
        refine_response = model.generate_content(user_question)
        
        raw_content = refine_response.text or "{}"
        print(f"[AI Analysis] {raw_content}")
        
        try:
            parsed_result = json.loads(raw_content)
            refined_query = parsed_result.get("refined_query", user_question)
            dynamic_weight = parsed_result.get("recency_weight", 0.1)
            start_date = parsed_result.get("start_date")
            end_date = parsed_result.get("end_date")
            ai_num_diaries = max(5, min(MAX_NUM_DIARIES, 
                                                       parsed_result.get("num_diaries", DEFAULT_NUM_RESULTS)))
        except json.JSONDecodeError:
            print("[Warning] Failed to parse JSON, using fallback values")
            refined_query = user_question
            dynamic_weight = 0.1
            start_date = None
            end_date = None
            ai_num_diaries = DEFAULT_NUM_RESULTS
        
        print(f"[Refined Query] {refined_query}")
        print(f"[Time Range] {start_date} ~ {end_date}")
        print(f"[Recency Weight] {dynamic_weight}")
        print(f"[Num Diaries] {ai_num_diaries} (AI decision)")
        
        temp_dir = os.path.join(self.project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        self._save_debug_file(
            os.path.join(temp_dir, "refined_query.txt"),
            f"Original: {user_question}\n\nRefined: {refined_query}\n\n"
            f"Time Range: {start_date} ~ {end_date}\n\nRecency Weight: {dynamic_weight}"
        )
        
        print(f"\n[Step 2] Searching for relevant diaries "
              f"(top {ai_num_diaries}, weight={dynamic_weight:.2f})...")
        query_results = self.query(
            refined_query, ai_num_diaries, 
            recency_weight=dynamic_weight,
            start_date=start_date, 
            end_date=end_date
        )
        
        self._save_debug_file(
            os.path.join(temp_dir, "query_result.json"),
            json.dumps(query_results, ensure_ascii=False, indent=4)
        )
        
        print("\n[Step 3] Building context and asking Gemini...")
        sys_instruction = "根据context（用户写的日记）精确回答问题，如果context不包含答案则如实回答搜索不到答案，不可杜撰"
        
        answer_model = genai.GenerativeModel(
             model_name=CHAT_MODEL,
             system_instruction=sys_instruction
        )
        
        prompt_parts = []
        for result in query_results:
            diary_context = f"日期: {result['date']}\n标题: {result['title']}\n内容:\n{result['content']}"
            prompt_parts.append(diary_context)
            
        prompt_parts.append("\n问题：" + user_question)
        
        print("\n[Answer]")
        print("-" * 60)
        
        stream = answer_model.generate_content(prompt_parts, stream=True)
        
        full_answer = ""
        for chunk in stream:
            if chunk.text:
                content = chunk.text
                print(content, end='', flush=True)
                full_answer += content
        
        print("\n" + "-" * 60)
        
        metadata = {
            "original_question": user_question,
            "refined_query": refined_query,
            "start_date": start_date,
            "end_date": end_date,
            "recency_weight": dynamic_weight,
            "num_diaries_requested": ai_num_diaries,
            "num_diaries_retrieved": len(query_results),
            "diary_dates": [r["date"] for r in query_results],
            "answer": full_answer
        }
        
        self._save_debug_file(
            os.path.join(temp_dir, "question_answer.json"),
            json.dumps(metadata, ensure_ascii=False, indent=4)
        )
        
        print(f"\n[Saved] Debug files saved to {temp_dir}")
        
        return full_answer
    
    def _save_debug_file(self, file_path: str, content: str) -> None:
        """Save debug information to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def run_test_cases(server: QueryServer) -> None:
    """Run test cases for the QueryServer."""
    print("=" * 60)
    print("Running Test Cases")
    print("=" * 60)
    
    test_queries = [
        ("学习", 3),
        ("心情", 3),
        ("朋友", 3)
    ]
    
    for query, num in test_queries:
        print(f"\n[Test] Query: '{query}', Num: {num}")
        print("-" * 60)
        results = server.query(query, num)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Date: {result['date']}")
            print(f"   Title: {result['title']}")
            print(f"   Content Preview: {result['content'][:100]}...")
        
        print("\n" + "=" * 60)


def test_recency_weight(server: QueryServer) -> None:
    """Test if recency_weight actually affects results."""
    print("\n" + "=" * 80)
    print("RECENCY WEIGHT TEST - Comparing different weight settings")
    print("=" * 80)
    
    test_query = "学习"
    num_results = 10
    
    # Test with different recency weights
    weights = [0.0, 0.15, 0.5, 1.0]
    
    print(f"\nQuery: '{test_query}' | Results: {num_results}\n")
    
    all_results = {}
    for weight in weights:
        print(f"\n{'='*80}")
        print(f"Testing with recency_weight = {weight}")
        print(f"{'='*80}")
        
        results = server.query(test_query, num_results, recency_weight=weight)
        all_results[weight] = results
        
        print(f"\n{'Rank':<6} {'Date':<20} {'Title':<30} {'Score':<8} {'Semantic':<10} {'Freshness':<10}")
        print("-" * 94)
        
        for i, result in enumerate(results, 1):
            date = result['date']
            title = result['title'][:28]
            score = result.get('score', 0)
            semantic = result.get('semantic', 0)
            freshness = result.get('freshness', 0)
            
            dt_obj = datetime.strptime(date, DATE_FORMAT)
            days_ago = (datetime.now() - dt_obj).days
            
            print(f"{i:<6} {date:<20} {title:<30} {score:.4f}   {semantic:.4f}     {freshness:.4f}    ({days_ago} days ago)")
    
    print(f"\n\n{'='*80}")
    print("COMPARISON: Weight 0.0 vs Weight 0.5")
    print("=" * 80)
    
    w0_dates = [r['date'] for r in all_results[0.0][:5]]
    w5_dates = [r['date'] for r in all_results[0.5][:5]]
    
    print("\nTop 5 with weight=0.0 (semantic only):")
    for i, date in enumerate(w0_dates, 1):
        print(f"  {i}. {date}")
    
    print("\nTop 5 with weight=0.5 (with recency boost):")
    for i, date in enumerate(w5_dates, 1):
        print(f"  {i}. {date}")
    
    if w0_dates != w5_dates:
        print(f"\n✅ PASS: Recency weight IS affecting results (order changed)")
        print(f"   Changed positions: {sum(1 for i, d in enumerate(w0_dates) if d != w5_dates[i])}/5")
    else:
        print(f"\n❌ FAIL: Recency weight NOT affecting results (order identical)")
    
    print(f"\n\n{'='*80}")
    print("SCORE ANALYSIS")
    print("=" * 80)
    
    for weight in [0.0, 0.5]:
        results = all_results[weight][:5]
        print(f"\nWith weight={weight}:")
        print(f"  Semantic score range: {min(r['semantic'] for r in results):.4f} - {max(r['semantic'] for r in results):.4f}")
        print(f"  Freshness score range: {min(r['freshness'] for r in results):.4f} - {max(r['freshness'] for r in results):.4f}")
        print(f"  Final score range: {min(r['score'] for r in results):.4f} - {max(r['score'] for r in results):.4f}")
        
        # Show how much recency contributes
        if weight > 0:
            avg_freshness_contrib = weight * sum(r['freshness'] for r in results) / len(results)
            print(f"  Average recency contribution: {avg_freshness_contrib:.4f}")
    
    print("\n" + "=" * 80)


def interactive_mode(server: QueryServer) -> None:
    """Interactive mode for manual query input."""
    print("\n" + "=" * 60)
    print("Interactive Query Mode")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 60)
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            num_input = input("Number of results (default 5): ").strip()
            num = int(num_input) if num_input else 5
        except ValueError:
            print("Invalid number, using default (5)")
            num = 5
        
        print(f"\n[Query] '{query}' (top {num} results)")
        print("-" * 60)
        
        results = server.query(query, num)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Date: {result['date']}")
            print(f"   Title: {result['title']}")
            print(f"   Content Preview: {result['content'][:150]}...")
        
        temp_dir = os.path.join(server.project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "query_result.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\n[Saved] Results written to: {output_path}")
        print("=" * 60)


def question_mode(server: QueryServer) -> None:
    """Question answering mode with RAG."""
    print("\n" + "=" * 60)
    print("Question Answering Mode (RAG)")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 60)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        try:
            server.question(question)
        except Exception as e:
            print(f"\n[Error] {e}")
        
        print("\n" + "=" * 60)


if __name__ == '__main__':
    try:
        server = QueryServer()
        
        print("\nQuery Server Options:")
        print("1. Run test cases")
        print("2. Interactive query mode")
        print("3. Question answering mode (RAG)")
        print("4. Test recency weight functionality")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_test_cases(server)
        elif choice == '2':
            interactive_mode(server)
        elif choice == '3':
            question_mode(server)
        elif choice == '4':
            test_recency_weight(server)
        else:
            print("Invalid choice. Running recency weight test by default.")
            test_recency_weight(server)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input('Press Enter to continue...')
