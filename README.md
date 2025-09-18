# nexresume

Resume-JD matching and scoring pipeline.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here   # optional
```

## Run

- Put resumes as .txt files in `resumes/`.
- Edit `job_description.yml` as needed.
- Execute:

```bash
python main.py
```

If `OPENAI_API_KEY` is set, the script will call the LLM and save JSON reports in `reports/`. Without a key, it will do a dry run and list which resumes would be processed.

### Per-project env (.env)

- Create `.env` in the project root:

```
OPENAI_API_KEY=your_key_here
```

- The script auto-loads `.env` via `python-dotenv`.

## Project Structure

```
nexresume/
  job_description.yml
  resumes/
  reports/
  main.py
  utils.py
  requirements.txt
  README.md
```

## Notes

- Output JSON schema includes: matched_required_skills, missing_required_skills, matched_optional_skills, education_match, experience_match, keywords_matched, soft_skills_match, resume_summary, match_score, city_tier_match, longest_tenure_months, final_score.
- Post-processing adjusts final_score using city tier, tenure bonus, and job gap penalty.
