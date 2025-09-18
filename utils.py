import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import yaml


@dataclass
class JobDescription:
    job_title: str
    required_skills: List[str]
    optional_skills: List[str]
    city_tier: int
    minimum_experience_years: Optional[float]
    maximum_job_gap_months: Optional[int]
    education_required: Optional[str]
    notes: Optional[str]


def _normalize_skill_list(skills: Any) -> List[str]:
    if skills is None:
        return []
    if isinstance(skills, str):
        parts = re.split(r",|/|;|\n", skills)
        return [s.strip().lower() for s in parts if s and s.strip()]
    if isinstance(skills, list):
        return [str(s).strip().lower() for s in skills if str(s).strip()]
    raise ValueError("Invalid skills format; expected list or string")


def load_job_description(path: str) -> JobDescription:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Support legacy/OG field names by mapping them to the expected schema
    def coerce_city_tier(value: Any) -> Any:
        if isinstance(value, str):
            m = re.search(r"(\d)", value)
            if m:
                return int(m.group(1))
        return value

    def first_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, (str, int, float)):
            return str(value)
        return None

    mapped: Dict[str, Any] = dict(data)
    # Title
    mapped.setdefault("job_title", data.get("Job_Title") or data.get("Job_title"))
    # Skills
    mapped.setdefault("required_skills", data.get("Required_Skills"))
    mapped.setdefault("optional_skills", data.get("Optional_Skills"))
    # Experience
    mapped.setdefault("minimum_experience_years", data.get("Minimum_Experience_Years"))
    # City tier may be like "Tier-2"
    if "city_tier" not in mapped and "City_Tier" in data:
        mapped["city_tier"] = coerce_city_tier(data.get("City_Tier"))
    # Job gap
    mapped.setdefault("maximum_job_gap_months", data.get("Maximum_Job_Gap_Months"))
    # Education: prefer explicit education_required if present, else map Educational_Qualification (first item)
    if "education_required" not in mapped:
        mapped["education_required"] = first_str(data.get("Educational_Qualification"))
    # Notes: append Job_Summary and Additional_Notes if present
    extra_notes_parts: List[str] = []
    if data.get("Job_Summary"):
        extra_notes_parts.append(str(data.get("Job_Summary")))
    if data.get("Additional_Notes"):
        extra_notes_parts.append(str(data.get("Additional_Notes")))
    if extra_notes_parts:
        existing = str(mapped.get("notes", "")).strip()
        combined = "\n".join([p.strip() for p in extra_notes_parts if p]).strip()
        mapped["notes"] = f"{existing}\n{combined}".strip()
    # Optional Certifications appended to notes if present
    if data.get("Optional_Certifications"):
        certs = data.get("Optional_Certifications")
        if isinstance(certs, list):
            certs_str = ", ".join(str(c).strip() for c in certs if str(c).strip())
        else:
            certs_str = str(certs).strip()
        prev = str(mapped.get("notes", "")).strip()
        mapped["notes"] = (prev + ("\n" if prev else "") + f"Optional Certifications: {certs_str}").strip()
    # Location is not used for scoring but allowed; ignore or include in notes
    if data.get("Location"):
        loc = str(data.get("Location")).strip()
        prev = str(mapped.get("notes", "")).strip()
        mapped["notes"] = (prev + ("\n" if prev else "") + f"Location: {loc}").strip()

    data = mapped

    # Validation of mandatory fields
    mandatory_fields = ["job_title", "required_skills", "city_tier"]
    missing = [k for k in mandatory_fields if k not in data or data[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing mandatory JD fields: {', '.join(missing)}")

    # Normalization
    job_title = str(data["job_title"]).strip()
    required_skills = _normalize_skill_list(data.get("required_skills"))
    optional_skills = _normalize_skill_list(data.get("optional_skills"))

    city_tier_raw = data.get("city_tier")
    try:
        city_tier = int(city_tier_raw)
        if city_tier not in (1, 2, 3):
            raise ValueError
    except Exception as exc:
        raise ValueError("city_tier must be 1, 2, or 3") from exc

    minimum_experience_years = None
    if data.get("minimum_experience_years") is not None:
        minimum_experience_years = float(data["minimum_experience_years"])

    maximum_job_gap_months = None
    if data.get("maximum_job_gap_months") is not None:
        maximum_job_gap_months = int(data["maximum_job_gap_months"])  # noqa: N816

    education_required = None
    if data.get("education_required"):
        education_required = str(data["education_required"]).strip()

    notes = None
    if data.get("notes"):
        notes = str(data["notes"]).strip()

    return JobDescription(
        job_title=job_title,
        required_skills=required_skills,
        optional_skills=optional_skills,
        city_tier=city_tier,
        minimum_experience_years=minimum_experience_years,
        maximum_job_gap_months=maximum_job_gap_months,
        education_required=education_required,
        notes=notes,
    )


# ------------------------------ Prompt Utilities ------------------------------
def get_output_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "matched_required_skills",
            "missing_required_skills",
            "matched_optional_skills",
            "education_match",
            "experience_match",
            "keywords_matched",
            "soft_skills_match",
            "resume_summary",
            "match_score",
            "city_tier_match",
            "longest_tenure_months",
            "final_score",
        ],
        "properties": {
            "matched_required_skills": {"type": "array", "items": {"type": "string"}},
            "missing_required_skills": {"type": "array", "items": {"type": "string"}},
            "matched_optional_skills": {"type": "array", "items": {"type": "string"}},
            "education_match": {"type": "boolean"},
            "experience_match": {"type": "boolean"},
            "keywords_matched": {"type": "array", "items": {"type": "string"}},
            "soft_skills_match": {"type": "array", "items": {"type": "string"}},
            "resume_summary": {"type": "string"},
            "match_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "city_tier_match": {"type": "boolean"},
            "longest_tenure_months": {"type": "integer", "minimum": 0},
            "final_score": {"type": "integer", "minimum": 0, "maximum": 100},
        },
    }


def _skills_to_bullets(title: str, skills: List[str]) -> str:
    if not skills:
        return f"- {title}: (none)"
    return "\n".join([f"- {title}:"] + [f"  - {s}" for s in skills])


def build_system_prompt(jd: JobDescription) -> str:
    schema = get_output_json_schema()
    schema_brief = (
        "Return ONLY valid JSON with the following keys: "
        "matched_required_skills, missing_required_skills, matched_optional_skills, "
        "education_match, experience_match, keywords_matched, soft_skills_match, "
        "resume_summary, match_score, city_tier_match, longest_tenure_months, final_score."
    )
    parts: List[str] = [
        "You are a meticulous ATS evaluator.",
        "Compare the job description with the resume and output STRICT JSON only.",
        schema_brief,
        "No prose, no markdown, no explanations—only a single JSON object.",
        "Scoring guidance:",
        "- City tier weightage: Tier-3 > Tier-2 > Tier-1.",
        "- Bonus for longest tenure (job stability).",
        "- Penalize long job gaps beyond Maximum_Job_Gap_Months.",
        "- match_score is 0–1; final_score is 0–100 and must reflect rules.",
        "Job Description Summary:",
        f"- Title: {jd.job_title}",
        _skills_to_bullets("Required skills", jd.required_skills),
        _skills_to_bullets("Optional skills", jd.optional_skills),
        f"- City tier: {jd.city_tier}",
        f"- Minimum experience (years): {jd.minimum_experience_years}",
        f"- Max job gap (months): {jd.maximum_job_gap_months}",
        f"- Education required: {jd.education_required}",
    ]
    return "\n".join(parts)


def build_user_prompt(resume_text: str) -> str:
    return (
        "Resume text provided below. Assess against the job description in the system message.\n"
        "Output a SINGLE JSON object only.\n\n"
        "Resume:\n" + resume_text
    )


def _extract_json_candidate(text: str) -> Optional[str]:
    try:
        obj = json.loads(text)
        return json.dumps(obj)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def parse_llm_json_or_raise(text: str) -> Dict[str, Any]:
    candidate = _extract_json_candidate(text)
    if not candidate:
        raise ValueError("LLM did not return JSON")
    try:
        return json.loads(candidate)
    except Exception as exc:
        raise ValueError("Failed to parse LLM JSON") from exc


def call_openai_with_retries(system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            return parse_llm_json_or_raise(content)
        except Exception as exc:  # retry on any error or bad JSON
            last_error = exc
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


# ------------------------------ Reports Utilities ------------------------------
def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "candidate"


def extract_candidate_name(resume_text: str, filename: Optional[str] = None) -> str:
    # Try simple pattern at the very top: a name-like line
    first_lines = "\n".join(resume_text.splitlines()[:5])
    m = re.search(r"([A-Z][a-z]+\s+[A-Z][a-zA-Z\-']+)", first_lines)
    if m:
        return m.group(1).strip()
    if filename:
        base = os.path.basename(filename)
        base = re.sub(r"\.[^.]+$", "", base)
        base = base.replace("_", " ").replace("-", " ")
        return base.strip() or "Candidate"
    return "Candidate"


def save_json_report(report: Dict[str, Any], candidate_name: str, reports_dir: str = "reports") -> str:
    os.makedirs(reports_dir, exist_ok=True)
    safe_name = _slugify(candidate_name)
    out_path = os.path.join(reports_dir, f"{safe_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def read_resume_text_from_txt(path: str, max_chars: int = 15000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if len(content) > max_chars:
        content = content[:max_chars]
    return content


def read_resume_text_from_pdf(path: str, max_chars: int = 15000) -> str:
    try:
        import pdfplumber
    except Exception as exc:
        raise RuntimeError("pdfplumber is required to read PDFs. Install dependencies.") from exc
    texts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t:
                texts.append(t)
    content = "\n".join(texts)
    if len(content) > max_chars:
        content = content[:max_chars]
    return content


def iterate_resume_texts(resumes_dir: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not os.path.isdir(resumes_dir):
        return entries
    for name in os.listdir(resumes_dir):
        path = os.path.join(resumes_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith(".txt"):
            try:
                text = read_resume_text_from_txt(path)
                entries.append({"filename": path, "text": text})
            except Exception:
                # skip unreadable files
                continue
        elif lower.endswith(".pdf"):
            try:
                text = read_resume_text_from_pdf(path)
                entries.append({"filename": path, "text": text})
            except Exception:
                continue
    return entries


# ------------------------------ Scoring Utilities ------------------------------
def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def apply_postprocessing_scoring(jd: JobDescription, llm_result: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(llm_result)

    base_score = float(result.get("final_score", 0))
    longest_tenure_months = int(result.get("longest_tenure_months", 0) or 0)
    # Use deterministic city tier mapping instead of LLM guess
    city_tier_detected = _detect_city_tier_from_resume_text(result.get("resume_summary", ""))
    city_tier_match = city_tier_detected == jd.city_tier
    result["city_tier_detected"] = city_tier_detected
    result["city_tier_match"] = bool(city_tier_match)
    job_gap_months = int(result.get("job_gap_months", 0) or 0)  # optional field if present

    adjustments: Dict[str, Any] = {
        "base_final_score": base_score,
        "city_tier_bonus": 0,
        "tenure_bonus": 0,
        "gap_penalty": 0,
    }

    # City tier weightage: Tier-3 > Tier-2 > Tier-1, applied when matched
    if city_tier_match:
        tier_bonus_map = {1: 2, 2: 4, 3: 6}
        adjustments["city_tier_bonus"] = tier_bonus_map.get(jd.city_tier, 0)

    # Longest tenure bonus: +2 points per full year, capped at +10
    tenure_years = longest_tenure_months / 12.0
    adjustments["tenure_bonus"] = clamp(tenure_years * 2.0, 0, 10)

    # Penalize job gaps beyond Maximum_Job_Gap_Months if both known
    if jd.maximum_job_gap_months is not None and job_gap_months:
        over = max(0, job_gap_months - int(jd.maximum_job_gap_months))
        # 1 point per extra month, capped at -15
        adjustments["gap_penalty"] = -min(over, 15)

    new_final = base_score + adjustments["city_tier_bonus"] + adjustments["tenure_bonus"] + adjustments["gap_penalty"]
    new_final = int(round(clamp(new_final, 0, 100)))

    result["final_score"] = new_final
    result["postprocessing_details"] = adjustments
    return result


def _load_city_tiers() -> Dict[str, int]:
    """Load city to tier mapping from city_tiers.yml"""
    try:
        with open("city_tiers.yml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {city.lower(): tier for city, tier in data.items()}
    except Exception:
        return {}


def _detect_city_tier_from_resume_text(resume_text: str) -> int:
    """Extract city from resume text and map to tier using deterministic mapping"""
    if not resume_text:
        return 2  # default
    
    city_tiers = _load_city_tiers()
    text_lower = resume_text.lower()
    
    # Look for city mentions in the text
    for city, tier in city_tiers.items():
        if city in text_lower:
            return tier
    
    # If no city found, return default tier-2
    return 2

