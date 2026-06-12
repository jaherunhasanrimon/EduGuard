"""
EduGuard — Rule-Based Intervention Engine
Maps student features and SHAP factors → prioritized advisor action items.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import FEATURE_DISPLAY_NAMES


# ─── Cause Tags (for Student List table pills) ────────────────────────────────

def get_cause_tags(student_row: dict, top_factors: list) -> list:
    """
    Return a short list of cause-tag strings for inline display in the
    student table (e.g. 'Low GPA', 'Fee Overdue').
    """
    tags = []
    r = student_row

    sem2_approved = float(r.get("Curricular units 2nd sem (approved)", 6))
    sem1_approved = float(r.get("Curricular units 1st sem (approved)", 6))
    sem2_grade    = float(r.get("Curricular units 2nd sem (grade)", 12))
    tuition_ok    = int(r.get("Tuition fees up to date", 1))
    is_debtor     = int(r.get("Debtor", 0))
    has_scholarship = int(r.get("Scholarship holder", 0))
    age           = float(r.get("Age at enrollment", 20))

    if tuition_ok == 0:
        tags.append("Fee Overdue")
    if is_debtor == 1:
        tags.append("Debtor")
    if sem2_approved == 0 and float(r.get("Curricular units 2nd sem (enrolled)", 1)) > 0:
        tags.append("Failing All")
    elif sem2_approved < 3:
        tags.append("Low Passes")
    if 0 < sem2_grade < 10:
        tags.append("Low GPA")
    if sem1_approved < 3 and sem2_approved < 3:
        tags.append("Consecutive Fails")
    if has_scholarship == 0 and is_debtor == 1:
        tags.append("No Scholarship")
    if age > 30:
        tags.append("Mature Student")

    # Fill remaining slots from top SHAP risk factors
    if len(tags) < 2 and top_factors:
        for f in top_factors:
            if f["shap_value"] > 0 and f["display_name"] not in tags:
                short = f["display_name"][:22]
                tags.append(short)
            if len(tags) >= 3:
                break

    return tags[:4]  # Maximum 4 tags per student


# ─── Intervention Actions ─────────────────────────────────────────────────────

def get_interventions(student_row: dict, top_factors: list) -> list:
    """
    Return a prioritised list of intervention action dicts for an individual
    student's deep-dive page.

    Each dict has keys:
        priority  : 'HIGH' | 'MEDIUM' | 'LOW'
        icon      : emoji string
        title     : short action title
        action    : full advisor-facing description
        trigger   : the feature/rule that fired this intervention
    """
    interventions = []
    r = student_row

    sem2_approved = float(r.get("Curricular units 2nd sem (approved)", 6))
    sem1_approved = float(r.get("Curricular units 1st sem (approved)", 6))
    sem2_enrolled = float(r.get("Curricular units 2nd sem (enrolled)", 6))
    sem1_enrolled = float(r.get("Curricular units 1st sem (enrolled)", 6))
    sem2_grade    = float(r.get("Curricular units 2nd sem (grade)", 12))
    sem1_grade    = float(r.get("Curricular units 1st sem (grade)", 12))
    tuition_ok    = int(r.get("Tuition fees up to date", 1))
    is_debtor     = int(r.get("Debtor", 0))
    has_scholarship = int(r.get("Scholarship holder", 0))
    is_international = int(r.get("International", 0))
    age           = float(r.get("Age at enrollment", 20))
    special_needs = int(r.get("Educational special needs", 0))

    # Rule 1 — Complete academic failure (both semesters)
    if sem2_approved == 0 and sem1_approved == 0 and (sem2_enrolled > 0 or sem1_enrolled > 0):
        interventions.append({
            "priority": "HIGH",
            "icon": "🚨",
            "title": "Urgent — Academic Failure",
            "action": (
                "Student has passed ZERO curricular units across two consecutive semesters. "
                "Escalate immediately to the Dean's office. A mandatory in-person meeting "
                "with the student and their academic advisor is required before next semester registration."
            ),
            "trigger": "Zero Units Passed (2 semesters)",
        })

    # Rule 2 — Low passes this semester
    elif sem2_approved < 3 and sem2_enrolled > 0:
        interventions.append({
            "priority": "HIGH",
            "icon": "📚",
            "title": "Academic Counseling — Low Unit Pass Rate",
            "action": (
                f"Student passed only {int(sem2_approved)} out of {int(sem2_enrolled)} "
                f"enrolled units this semester. Schedule an academic review meeting to assess "
                f"workload manageability, identify struggling subjects, and create a recovery plan."
            ),
            "trigger": "Low Courses Passed",
        })

    # Rule 3 — Fee arrears
    if tuition_ok == 0:
        interventions.append({
            "priority": "HIGH",
            "icon": "💰",
            "title": "Financial Aid Referral — Fee Overdue",
            "action": (
                "Tuition fees are not current. Contact the student immediately and refer them "
                "to the Financial Aid Office to explore payment plans, emergency bursary funds, "
                "or deferred payment options. Fee arrears are a strong predictor of dropout."
            ),
            "trigger": "Tuition Fees Overdue",
        })

    # Rule 4 — Outstanding debt
    if is_debtor == 1 and tuition_ok == 1:
        interventions.append({
            "priority": "MEDIUM",
            "icon": "🏦",
            "title": "Debt Counseling Referral",
            "action": (
                "Student has outstanding financial obligations. Refer to the Student Welfare "
                "Office for debt counseling services and review eligibility for scholarships, "
                "grants, or income-contingent repayment schemes."
            ),
            "trigger": "Outstanding Debt",
        })

    # Rule 5 — Low academic grades
    if 0 < sem2_grade < 10:
        interventions.append({
            "priority": "MEDIUM",
            "icon": "📝",
            "title": "Tutoring & Supplemental Instruction",
            "action": (
                f"Semester 2 average grade is {sem2_grade:.1f}/20, well below the passing "
                f"threshold. Enrol student in the peer tutoring programme and identify the "
                f"2–3 specific courses with the lowest performance for targeted support."
            ),
            "trigger": "Low Academic Grade",
        })

    # Rule 6 — Scholarship gap + debt
    if has_scholarship == 0 and is_debtor == 1:
        interventions.append({
            "priority": "MEDIUM",
            "icon": "🎓",
            "title": "Scholarship Eligibility Review",
            "action": (
                "Student holds no scholarship but has financial difficulties. Review eligibility "
                "for merit-based, need-based, or departmental scholarships. A bursary application "
                "submitted this semester could significantly reduce dropout risk."
            ),
            "trigger": "No Scholarship + Debt",
        })

    # Rule 7 — Mature student
    if age > 30:
        interventions.append({
            "priority": "MEDIUM",
            "icon": "⚖️",
            "title": "Work-Life Balance Assessment",
            "action": (
                f"Student enrolled at age {int(age)}, suggesting potential competing commitments "
                f"(employment, family responsibilities). Discuss flexible scheduling, part-time "
                f"enrollment options, or remote/hybrid learning pathways."
            ),
            "trigger": "Mature-Age Student",
        })

    # Rule 8 — Special educational needs
    if special_needs == 1:
        interventions.append({
            "priority": "MEDIUM",
            "icon": "♿",
            "title": "Disability & Accessibility Support",
            "action": (
                "Student has registered special educational needs. Confirm all required "
                "accommodations (extended exam time, accessible materials, etc.) are in place "
                "and schedule a check-in with the Disability Support Officer."
            ),
            "trigger": "Special Educational Needs",
        })

    # Rule 9 — International student
    if is_international == 1:
        interventions.append({
            "priority": "LOW",
            "icon": "🌍",
            "title": "International Student Integration Check",
            "action": (
                "International students face unique adjustment and cultural challenges. Connect "
                "with the International Student Office for cultural integration support, language "
                "assistance, and visa/accommodation guidance."
            ),
            "trigger": "International Student",
        })

    # Fallback — no specific rules fired
    if not interventions:
        top_feature = top_factors[0] if top_factors else None
        label = top_feature["display_name"] if top_feature else "Academic Performance"
        interventions.append({
            "priority": "MEDIUM",
            "icon": "⚠️",
            "title": "Proactive Check-In Recommended",
            "action": (
                f"The primary risk factor identified is '{label}'. Although no critical warning "
                f"triggers have fired, the model has flagged this student as elevated risk. "
                f"A short, informal check-in meeting is recommended to assess wellbeing and "
                f"identify any emerging issues early."
            ),
            "trigger": label,
        })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    interventions.sort(key=lambda x: priority_order[x["priority"]])

    return interventions[:5]  # Maximum 5 interventions
