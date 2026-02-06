import asyncio
import csv
import dataclasses
import json
import os
import queue
import sys
import threading
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
from loguru import logger

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.prompt import (
    get_system_prompt,
    get_tools_api_description,
)
from src.agent.run import run_single_query
from src.agent.tools import _default_tools
from src.evaluation.data_loader import (
    WideSearchDataLoaderHF,
    WideSearchQuery,
    WideSearchResponse,
)
from src.evaluation.evaluation import EvaluationResult, evaluate_single_query
from src.utils.llm import get_model_config_no_key

logger.remove()
logger.add(sys.stderr, level="INFO")


class SingleTask:
    def __init__(
        self,
        query: WideSearchQuery,
        model_config_name: str,
        trial_idx: int = 1,
        eval_model_config_name: str = "openai/gpt-5.2",
        tools: dict = _default_tools,
    ):
        self.query = query
        self.trial_idx = trial_idx
        self.model_config_name = model_config_name
        self.eval_model_config_name = eval_model_config_name
        self.tools = tools

    async def infer(self):
        logger.info(f"infer start, instance_id: {self.query.instance_id}")
        start_time = time.time()
        tools = self.tools
        system_prompt = get_system_prompt(self.query.language)

        tools_desc = get_tools_api_description(self.query.language, list(tools.keys()))
        run_result = await run_single_query(
            query=self.query.query,
            agent_name=f"{self.query.instance_id}_{self.trial_idx}",
            model_config_name=self.model_config_name,
            tools=tools,
            system_prompt=system_prompt,
            tools_desc=tools_desc,
            collect_token_usage=True,
        )
        messages, stats = run_result
        response = "NULL"
        try:
            response = messages[-1]["content"]["content"]
        except Exception:
            response = messages[-1]["content"]

        messages = _sanitize_messages_for_save(messages)

        assistant_message_count = 0
        tool_call_counts = {
            "web_search_chinese": 0,
            "web_search_internationl": 0,
            "get_content": 0,
        }
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            assistant_message_count += 1
            content = m.get("content")
            if not isinstance(content, dict):
                continue
            tool_calls = content.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("tool_name")
                if name in tool_call_counts:
                    tool_call_counts[name] += 1

        wide_search_response = WideSearchResponse(
            instance_id=self.query.instance_id,
            response=response,
            assistant_message_count=assistant_message_count,
            tool_call_counts=tool_call_counts,
            messages=messages,
            trial_idx=self.trial_idx,
        )
        end_time = time.time()
        logger.info(
            f"infer end, instance_id: {self.query.instance_id}, cost(s): {end_time - start_time:.2f}"
        )
        return wide_search_response

    def eval(self, response: WideSearchResponse | None):
        start_time = time.time()
        eval_result = evaluate_single_query(
            self.query,
            response,
            None,
            self.eval_model_config_name,
        )
        end_time = time.time()
        logger.info(
            f"eval end, instance_id: {self.query.instance_id}, cost(s): {end_time - start_time:.2f}"
        )
        return eval_result


def _append_jsonl(path: str, obj: dict) -> None:
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if v is not None}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _sanitize_messages_for_save(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") != "assistant":
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, dict):
            out.append(m)
            continue

        tcrs = content.get("tool_call_results")
        if not isinstance(tcrs, list):
            out.append(m)
            continue

        new_tcrs = []
        for tcr in tcrs:
            if not isinstance(tcr, dict):
                new_tcrs.append(tcr)
                continue
            tcr = dict(tcr)
            tcr.pop("extra", None)

            tcr_content = tcr.get("content")
            if isinstance(tcr_content, str):
                try:
                    tcr["content"] = json.loads(tcr_content)
                except Exception:
                    pass
            new_tcrs.append(tcr)

        new_content = dict(content)
        new_content["tool_call_results"] = new_tcrs
        new_m = dict(m)
        new_m["content"] = new_content
        out.append(new_m)

    return out


def _is_error_response(response: WideSearchResponse) -> bool:
    resp = str(response.response)
    # Check for patterns indicating an error
    error_patterns = [
        "[Runner] Exception during LLM completion:",
        "Too many errors have occurred.",
        "[Max Step] The tool has been used too many times."
    ]
    for pattern in error_patterns:
        if pattern in resp:
            return True
    return False


def _load_responses_map(
    response_file: str,
) -> dict[tuple[str, int], WideSearchResponse]:
    responses_map: dict[tuple[str, int], WideSearchResponse] = {}
    if not os.path.exists(response_file):
        return responses_map
    
    with open(response_file, "r", encoding="utf-8") as f:
        # Try reading line by line first (standard JSONL)
        lines = f.readlines()
        
    # Attempt to parse line by line
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            # Skip model_config metadata line if present
            if "model_config" in item:
                continue
            resp = WideSearchResponse(**item)
            trial_idx = int(resp.trial_idx or 0)
            responses_map[(resp.instance_id, trial_idx)] = resp
        except (json.JSONDecodeError, TypeError):
            # If line-by-line fails for a specific line, try more robust parsing
            # This handles the case where multiple pretty-printed JSON objects are concatenated
            decoder = json.JSONDecoder()
            content = line
            idx = 0
            while idx < len(content):
                # Skip whitespace
                while idx < len(content) and content[idx].isspace():
                    idx += 1
                if idx >= len(content):
                    break
                    
                try:
                    obj, end_idx = decoder.raw_decode(content, idx)
                    if isinstance(obj, dict) and "model_config" in obj:
                        idx = end_idx
                        continue
                    resp = WideSearchResponse(**obj)
                    trial_idx = int(resp.trial_idx or 0)
                    responses_map[(resp.instance_id, trial_idx)] = resp
                    idx = end_idx
                except (json.JSONDecodeError, TypeError):
                    break
                
    return responses_map


def _is_non_empty_response(response: WideSearchResponse) -> bool:
    resp = response.response
    if resp is None:
        return False
    resp = str(resp).strip()
    if not resp:
        return False
    if resp == "NULL":
        return False
    return True


def _load_scored_keys(result_file: str) -> set[tuple[str, int]]:
    scored: set[tuple[str, int]] = set()
    if not os.path.exists(result_file):
        return scored
    with open(result_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = (row.get("instance_id") or "").strip()
            if not instance_id:
                continue
            score_val = row.get("score")
            if score_val is None or str(score_val).strip() == "":
                continue
            trial_idx_raw = row.get("trial_idx")
            try:
                trial_idx = int(float(trial_idx_raw)) if trial_idx_raw else 0
            except Exception:
                trial_idx = 0
            scored.add((instance_id, trial_idx))
    return scored


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _calc_summary_from_csv(result_file: str, trial_num: int) -> dict:
    metrics = [
        "score",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
    ]

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"result_file {result_file} not found")

    id_to_trial_metrics: dict[str, dict[str, list[float]]] = {}
    with open(result_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row.get("instance_id")
            if not instance_id:
                continue
            if instance_id not in id_to_trial_metrics:
                id_to_trial_metrics[instance_id] = {m: [] for m in metrics}
            for m in metrics:
                if row.get(m) is None or row.get(m) == "":
                    continue
                id_to_trial_metrics[instance_id][m].append(float(row[m]))

    all_results = {m: [] for m in metrics}

    for iid, trial_metrics in id_to_trial_metrics.items():
        for m in metrics:
            values = trial_metrics[m]
            if not values or len(values) < trial_num:
                # If not enough trials, skip this instance for this metric
                logger.info(f"Skipping {m} for instance {iid}, not enough trials")
                raise ValueError(
                    f"Not enough trials for metric {m} on instance {iid}. "
                    f"Expected {trial_num}, got {len(values)}."
                )
            avg_n = float(np.mean(values))
            max_n = float(np.max(values))
            min_n = float(np.min(values))
            all_results[m].append({"avg_n": avg_n, "max_n": max_n, "min_n": min_n})

    # Aggregate over all instances
    summary = {}
    for m in metrics:
        vals = all_results[m]
        if not vals:
            continue
        summary[m] = {
            "avg_n": float(np.mean([v["avg_n"] for v in vals])),
            "max_n": float(np.mean([v["max_n"] for v in vals])),
            "min_n": float(np.mean([v["min_n"] for v in vals])),
        }
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/gpt-5.2",
        help="model config name",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["eval", "infer", "both"],
        help="stage to run",
    )

    parser.add_argument(
        "--response-file", type=str, default="", help="response file path (jsonl)"
    )
    parser.add_argument(
        "--result-file", type=str, default="", help="result file path (csv)"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="google/gemini-3-flash-preview",
        help="eval model config name",
    )
    parser.add_argument("--trial_num", type=int, default=1, help="trial num to run")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="only run first N tasks (after expanding trials). If omitted, run all.",
    )
    parser.add_argument(
        "--thread_num", type=int, default=30, help="thread num to run infer and eval"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["zh", "en"],
        help="filter tasks by language",
    )

    args = parser.parse_args()

    trial_num = args.trial_num
    model_config_name = args.model_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    response_file_provided = bool(args.response_file)
    result_file_provided = bool(args.result_file)

    response_file = args.response_file
    if not response_file:
        response_file = str(
            Path("data") / model_config_name / "response" / f"{timestamp}.jsonl"
        )
    result_file = args.result_file
    if not result_file:
        result_file = str(
            Path("data") / model_config_name / "eval" / f"{timestamp}.csv"
        )

    _ensure_parent_dir(response_file)
    _ensure_parent_dir(result_file)

    print(f"DEBUG: response_file path is: {os.path.abspath(response_file)}")
    summary_result_path = str(Path(result_file).parent / f"{timestamp}_summary.json")

    data_loader = WideSearchDataLoaderHF()

    instance_id_list = data_loader.get_instance_id_list()

    tasks = []

    tools = _default_tools

    for instance_id in instance_id_list:
        query = data_loader.load_query_by_instance_id(instance_id)

        if args.language and query.language != args.language:
            continue

        for trial_idx in range(trial_num):
            tasks.append(
                SingleTask(
                    query=deepcopy(query),
                    trial_idx=trial_idx,
                    model_config_name=model_config_name,
                    tools=tools,
                    eval_model_config_name=args.evaluator,
                )
            )

    if args.num_tasks is not None and args.num_tasks > 0:
        tasks = tasks[: args.num_tasks]

    logger.info(f"total task num: {len(tasks)}")
    if args.stage in ["infer", "both"]:
        infer_tasks = tasks
        if response_file_provided and os.path.exists(response_file):
            responses_map = _load_responses_map(response_file)
            done_keys = {
                k for k, v in responses_map.items() if _is_non_empty_response(v)
            }
            infer_tasks = [
                task
                for task in tasks
                if (task.query.instance_id, task.trial_idx) not in done_keys
            ]
            logger.info(
                f"resume infer: skip {len(tasks) - len(infer_tasks)}, run {len(infer_tasks)}"
            )

        if infer_tasks:
            write_queue: queue.Queue[object] = queue.Queue(maxsize=100)
            write_sentinel = object()
            write_lock = threading.Lock()
            stop_event = threading.Event()
            inflight_slots: threading.Semaphore = threading.Semaphore(args.thread_num)
            written_count_ref = [0]
            write_failed_count_ref = [0]

            def _writer() -> None:
                while True:
                    try:
                        item = write_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_event.is_set():
                            return
                        continue
                    try:
                        if item is write_sentinel:
                            return
                        if not isinstance(item, dict):
                            raise TypeError(f"writer expected dict, got {type(item)}")

                        instance_id = item.get("instance_id")
                        trial_idx = item.get("trial_idx")
                        
                        # Check if response is empty
                        resp_content = item.get("response")
                        if not resp_content or str(resp_content).strip() in ["", "NULL"]:
                            logger.warning(
                                f"infer skipped writing, response is empty, instance_id: {instance_id}, trial_idx: {trial_idx}"
                            )
                            continue

                        _append_jsonl(response_file, item)
                        with write_lock:
                            written_count_ref[0] += 1
                        logger.info(
                            f"infer written, instance_id: {instance_id}, trial_idx: {trial_idx}"
                        )
                    except Exception:
                        err = traceback.format_exc()
                        if isinstance(item, dict):
                            instance_id = item.get("instance_id")
                            trial_idx = item.get("trial_idx")
                        else:
                            instance_id = None
                            trial_idx = None

                        with write_lock:
                            write_failed_count_ref[0] += 1
                        
                        # Add specific failure reason if available
                        reason = "Unknown error"
                        if "item" in locals() and isinstance(item, dict):
                            resp_val = item.get("response")
                            if not resp_val or str(resp_val).strip() in ["", "NULL"]:
                                reason = "Empty response"
                            else:
                                reason = "File append error or JSON parsing error"
                        
                        logger.error(
                            f"infer write failed (Reason: {reason}), instance_id: {instance_id}, trial_idx: {trial_idx}, error: {err}"
                        )
                    finally:
                        if item is not write_sentinel and isinstance(item, dict):
                            inflight_slots.release()
                        write_queue.task_done()

            writer_thread = threading.Thread(target=_writer, daemon=True)
            writer_thread.start()

            executor = ThreadPoolExecutor(max_workers=args.thread_num)
            infer_iter = iter(infer_tasks)
            future_to_task: dict[object, SingleTask] = {}

            submit_limit = args.thread_num

            def _submit_one(task: SingleTask) -> None:
                inflight_slots.acquire()
                try:
                    fut = executor.submit(lambda task: asyncio.run(task.infer()), task)
                except Exception:
                    inflight_slots.release()
                    raise
                future_to_task[fut] = task

            while len(future_to_task) < submit_limit:
                try:
                    task = next(infer_iter)
                except StopIteration:
                    break
                _submit_one(task)
            enqueued_count = 0
            failed_count = 0
            interrupted = False
            try:
                while future_to_task:
                    done, _ = wait(
                        future_to_task.keys(), return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        task = future_to_task.pop(future)
                        try:
                            resp = future.result()
                            if _is_error_response(resp):
                                logger.warning(
                                    f"infer produced an error response, NOT writing: instance_id: {resp.instance_id}, trial_idx: {resp.trial_idx}, response: {resp.response[:100]}..."
                                )
                                inflight_slots.release()
                                continue

                            obj = dataclasses.asdict(resp)
                            print(
                                f"DEBUG: enqueue response for {resp.instance_id} (trial_idx={resp.trial_idx}) to {response_file}"
                            )
                            write_queue.put(obj)
                            enqueued_count += 1
                        except KeyboardInterrupt:
                            raise
                        except BaseException:
                            failed_count += 1
                            inflight_slots.release()
                            err = traceback.format_exc()
                            print(
                                f"DEBUG: infer failed (NOT written) for {task.query.instance_id} (trial_idx={task.trial_idx}): {err}"
                            )
                            logger.error(
                                f"infer failed (NOT written), instance_id: {task.query.instance_id}, trial_idx: {task.trial_idx}, error: {err}"
                            )

                    while len(future_to_task) < submit_limit:
                        try:
                            next_task = next(infer_iter)
                        except StopIteration:
                            break
                        _submit_one(next_task)
            except KeyboardInterrupt:
                interrupted = True
                logger.warning(
                    "KeyboardInterrupt received (Ctrl+C). Cancelling pending infer tasks and stopping writer."
                )
                stop_event.set()
                for fut in future_to_task:
                    fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                writer_thread.join(timeout=2)
                raise SystemExit(130)
            finally:
                if not interrupted:
                    executor.shutdown(wait=True)

            write_queue.put(write_sentinel)
            write_queue.join()
            writer_thread.join(timeout=10)
            with write_lock:
                _written_count = written_count_ref[0]
                _write_failed_count = write_failed_count_ref[0]

            logger.info(
                f"infer done, enqueued: {enqueued_count}, infer_failed_not_enqueued: {failed_count}, written: {_written_count}, write_failed: {_write_failed_count}"
            )

    if args.stage in ["eval", "both"]:
        eval_tasks = tasks
        if result_file_provided and os.path.exists(result_file):
            scored_keys = _load_scored_keys(result_file)
            eval_tasks = [
                task
                for task in tasks
                if (task.query.instance_id, task.trial_idx) not in scored_keys
            ]
            logger.info(
                f"resume eval: skip {len(tasks) - len(eval_tasks)}, run {len(eval_tasks)}"
            )

        if eval_tasks:
            responses_map = _load_responses_map(response_file)
            fieldnames = [
                "instance_id",
                "trial_idx",
                "score",
                "precision_by_row",
                "recall_by_row",
                "f1_by_row",
                "precision_by_item",
                "recall_by_item",
                "f1_by_item",
                "msg",
            ]

            file_exists = os.path.exists(result_file)
            file_non_empty = file_exists and os.path.getsize(result_file) > 0
            open_mode = "a" if (result_file_provided and file_exists) else "w"

            with open(result_file, open_mode, encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if open_mode == "w" or not file_non_empty:
                    writer.writeheader()

                with ThreadPoolExecutor(max_workers=args.thread_num) as executor:

                    def _eval_one(
                        task: SingleTask,
                    ) -> tuple[SingleTask, EvaluationResult]:
                        resp = responses_map.get(
                            (task.query.instance_id, task.trial_idx)
                        )
                        return task, task.eval(resp)

                    future_to_task = {
                        executor.submit(_eval_one, task): task for task in eval_tasks
                    }
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            task, result = future.result()
                            writer.writerow(
                                {
                                    "instance_id": result.instance_id,
                                    "trial_idx": task.trial_idx,
                                    "score": result.score,
                                    "precision_by_row": result.precision_by_row,
                                    "recall_by_row": result.recall_by_row,
                                    "f1_by_row": result.f1_by_row,
                                    "precision_by_item": result.precision_by_item,
                                    "recall_by_item": result.recall_by_item,
                                    "f1_by_item": result.f1_by_item,
                                    "msg": result.msg,
                                }
                            )
                            f.flush()
                            logger.info(
                                f"eval success, instance_id: {result.instance_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"eval task error for instance_id {task.query.instance_id}: {e}"
                            )

        summary = _calc_summary_from_csv(result_file=result_file, trial_num=trial_num)
        
        # Add evaluator config to summary
        evaluator_config = get_model_config_no_key(args.evaluator, is_evaluator=True)
        final_summary = {
            "evaluator_config": evaluator_config,
            "summary": summary
        }
        
        with open(summary_result_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
