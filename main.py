from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv

import os
from utils import (
    JobDescription,
    build_system_prompt,
    build_user_prompt,
    call_openai_with_retries,
    extract_candidate_name,
    iterate_resume_texts,
    get_output_json_schema,
    load_job_description,
    apply_postprocessing_scoring,
    save_json_report,
)


def main() -> None:
    load_dotenv()
    jd_path = Path("job_description.yml")
    if not jd_path.exists():
        raise SystemExit("job_description.yml not found. Please create it first.")

    jd: JobDescription = load_job_description(str(jd_path))
    # print(json.dumps({
    #     "job_title": jd.job_title,
    #     "required_skills": jd.required_skills,
    #     "optional_skills": jd.optional_skills,
    #     "city_tier": jd.city_tier,
    #     "minimum_experience_years": jd.minimum_experience_years,
    #     "maximum_job_gap_months": jd.maximum_job_gap_months,
    #     "education_required": jd.education_required,
    #     "notes": jd.notes,
    # }, indent=2, ensure_ascii=False))

    system_prompt = build_system_prompt(jd)
    # print("\n--- System Prompt Preview ---\n")
    # print(system_prompt)
    # print("\n--- Output Schema Keys ---\n")
    # print(sorted(get_output_json_schema()["properties"].keys()))

    # Batch process .txt resumes if present
    resumes_dir = "resumes"
    entries = iterate_resume_texts(resumes_dir)
    if entries:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = lambda x: x  # fallback if tqdm missing

        print(f"\n--- Processing {len(entries)} resume(s) from {resumes_dir} ---\n")
        for item in tqdm(entries):
            text = item["text"]
            filename = item["filename"]
            candidate = extract_candidate_name(filename)
            if os.getenv("OPENAI_API_KEY"):
                try:
                    user_prompt = build_user_prompt(text)
                    result = call_openai_with_retries(system_prompt, user_prompt)
                    result = apply_postprocessing_scoring(jd, result)
                    out_path = save_json_report(result, candidate)
                    print(f"Saved report: {out_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            else:
                print(f"Would process {filename} (set OPENAI_API_KEY to enable LLM)")


if __name__ == "__main__":
    main()


