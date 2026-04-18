"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic corpus of 2,500+ labelled business documents
across 8 categories and writes it to data/raw/documents.csv.

Usage
-----
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --n 3000 --output data/raw/custom.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Category templates
# Each entry: (category, [sentence templates])
# {NUM} is replaced by a random integer, {WORD} by a domain-specific noun.
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "Legal": [
        "This agreement is entered into as of {DATE} between the parties listed herein.",
        "The defendant is hereby ordered to pay damages not exceeding ${NUM},000.",
        "Pursuant to section {NUM} of the contract, all disputes shall be resolved by arbitration.",
        "The plaintiff alleges that the respondent breached the terms of the written agreement dated {DATE}.",
        "All intellectual property rights, including patents, trademarks, and copyrights, remain with the licensor.",
        "The indemnification clause requires the contractor to hold the client harmless from all third-party claims.",
        "Jurisdiction for any legal proceedings arising from this contract shall be the courts of the state of Delaware.",
        "The non-disclosure agreement prohibits the disclosure of proprietary information to any unauthorized party.",
        "Force majeure events, including acts of God and government actions, shall excuse performance obligations.",
        "The arbitration panel issued a binding ruling requiring compliance within {NUM} business days.",
        "Liability is limited to the total contract value not exceeding ${NUM},000 per incident.",
        "The court found sufficient grounds to grant a preliminary injunction pending full trial.",
        "This memorandum of understanding sets forth the preliminary terms agreed upon by both parties.",
        "Upon termination, all confidential materials must be returned or destroyed within {NUM} days.",
        "The statute of limitations for breach of contract claims is {NUM} years from the date of discovery.",
    ],
    "Medical": [
        "The patient presents with acute respiratory distress and elevated inflammatory markers.",
        "Blood pressure readings of {NUM}/{NUM} mmHg indicate stage {NUM} hypertension.",
        "MRI imaging revealed a {NUM} mm lesion in the left temporal lobe requiring further investigation.",
        "Post-operative care includes daily wound inspection and antibiotic prophylaxis for {NUM} days.",
        "The clinical trial enrolled {NUM} participants with a diagnosis of type {NUM} diabetes mellitus.",
        "Laboratory results indicate elevated creatinine levels suggesting early-stage renal insufficiency.",
        "The patient was administered {NUM} mg of metformin twice daily with meals.",
        "Echocardiogram findings show an ejection fraction of {NUM}%, indicating moderate cardiac dysfunction.",
        "Informed consent was obtained prior to the invasive procedure in accordance with hospital protocol.",
        "Follow-up appointment is scheduled in {NUM} weeks to reassess treatment response.",
        "The differential diagnosis includes viral pneumonia, pulmonary embolism, and congestive heart failure.",
        "Pathology report confirms adenocarcinoma with clear surgical margins at the resection site.",
        "Physical therapy sessions are recommended {NUM} times per week to restore functional mobility.",
        "The patient's BMI of {NUM} places them in the obese category requiring nutritional counselling.",
        "Vital signs on admission: temperature 38.{NUM}°C, heart rate {NUM} bpm, SpO2 {NUM}%.",
    ],
    "Financial": [
        "Q{NUM} revenue reached ${NUM}M, representing a {NUM}% year-over-year increase.",
        "Operating expenses totalled ${NUM},000, driven primarily by increased headcount and infrastructure costs.",
        "The portfolio achieved an annualised return of {NUM}% against a benchmark of {NUM}%.",
        "EBITDA margin improved by {NUM} basis points following the cost optimisation initiative.",
        "Cash and cash equivalents stood at ${NUM}M as of the end of the fiscal quarter.",
        "The board approved a share buyback programme of up to ${NUM}M over the next {NUM} months.",
        "Debt-to-equity ratio decreased from {NUM}.{NUM} to {NUM}.{NUM} following debt restructuring.",
        "Accounts receivable ageing report shows {NUM}% of balances outstanding beyond 90 days.",
        "Net present value of the investment is estimated at ${NUM},000 using a discount rate of {NUM}%.",
        "The audit committee reviewed the financial statements and found no material misstatements.",
        "Foreign exchange exposure was hedged using forward contracts totalling ${NUM}M notional value.",
        "Revenue recognition follows ASC 606 with performance obligations identified for each contract.",
        "Goodwill impairment testing was conducted per ASC 350 with no impairment charge recorded.",
        "The weighted average cost of capital is estimated at {NUM}.{NUM}%, used for investment appraisals.",
        "Working capital of ${NUM}M provides adequate liquidity for the next {NUM} months of operations.",
    ],
    "Technical": [
        "The microservice architecture is deployed on Kubernetes with {NUM} replicas per service.",
        "API latency p99 is {NUM}ms with an error rate below {NUM}% under peak load conditions.",
        "The database schema migration introduces {NUM} new indexes to improve query performance by {NUM}%.",
        "Continuous integration pipeline runs {NUM} unit tests and {NUM} integration tests on every pull request.",
        "Memory usage spiked to {NUM}GB during the batch processing job due to an unoptimised aggregation query.",
        "The load balancer is configured with a least-connections algorithm and health checks every {NUM} seconds.",
        "TLS {NUM}.{NUM} is enforced across all endpoints; weak cipher suites have been deprecated.",
        "Log aggregation is handled by the ELK stack with a retention policy of {NUM} days.",
        "The machine learning model achieved {NUM}% accuracy after {NUM} epochs of training on GPU.",
        "Cache hit ratio improved to {NUM}% following the introduction of Redis with a TTL of {NUM} seconds.",
        "Infrastructure is provisioned via Terraform with state stored in a remote S3 backend.",
        "The gRPC service exposes {NUM} methods with Protocol Buffer schemas versioned at v{NUM}.",
        "Static analysis flagged {NUM} critical and {NUM} medium severity issues in the latest code scan.",
        "Horizontal pod autoscaling triggers at {NUM}% CPU utilisation with a max replica count of {NUM}.",
        "The data pipeline ingests {NUM}K events per second with end-to-end latency under {NUM}ms.",
    ],
    "HR": [
        "The performance review cycle for Q{NUM} requires all managers to complete evaluations by {DATE}.",
        "Onboarding documentation must be completed within {NUM} business days of the employee start date.",
        "The compensation benchmarking study reveals a {NUM}% gap against the 75th percentile of the market.",
        "A total of {NUM} disciplinary actions were recorded in the past quarter, a {NUM}% decrease year-over-year.",
        "Employee engagement survey results show an overall satisfaction score of {NUM} out of 10.",
        "The recruitment pipeline currently has {NUM} open requisitions across {NUM} departments.",
        "Mandatory compliance training must be completed by all employees before {DATE}.",
        "The flexible working policy allows up to {NUM} remote working days per week.",
        "Redundancy consultation with the affected {NUM} employees commenced on {DATE}.",
        "Annual leave entitlement is {NUM} days, increasing to {NUM} days after {NUM} years of service.",
        "The learning & development budget for the fiscal year is ${NUM},000 per employee.",
        "Exit interview feedback indicated workload and career progression as the primary departure drivers.",
        "The diversity and inclusion report shows a {NUM}% improvement in leadership representation.",
        "Succession planning identified {NUM} high-potential employees for senior leadership roles.",
        "The updated grievance procedure ensures resolution within {NUM} working days of formal submission.",
    ],
    "Marketing": [
        "The campaign generated {NUM},000 impressions with a click-through rate of {NUM}.{NUM}%.",
        "A/B testing of subject lines resulted in a {NUM}% uplift in email open rates.",
        "The product launch achieved ${NUM}M in first-week sales, exceeding the forecast by {NUM}%.",
        "Social media following grew by {NUM}% following the influencer collaboration programme.",
        "Customer acquisition cost decreased to ${NUM} per lead after optimising paid search bids.",
        "The SEO initiative increased organic traffic by {NUM}% in {NUM} months.",
        "Net Promoter Score improved from {NUM} to {NUM} following the customer experience redesign.",
        "Content marketing produced {NUM} blog posts, {NUM} videos, and {NUM} whitepapers this quarter.",
        "The loyalty programme enrolled {NUM},000 new members, driving a {NUM}% increase in repeat purchases.",
        "Brand awareness surveys show aided recall of {NUM}% among the target demographic.",
        "Conversion rate optimisation testing increased checkout completions by {NUM}%.",
        "The referral programme contributed {NUM}% of new customer acquisitions this period.",
        "Programmatic advertising spend of ${NUM},000 delivered a return on ad spend of {NUM}x.",
        "Email list segmentation improved deliverability to {NUM}% and reduced unsubscribe rate to {NUM}%.",
        "Video content on the platform reached {NUM}M views with an average watch time of {NUM} minutes.",
    ],
    "Research": [
        "The study enrolled {NUM} participants across {NUM} research sites over a {NUM}-month period.",
        "Statistical analysis using ANOVA revealed a significant effect (F={NUM}.{NUM}, p<0.0{NUM}).",
        "The literature review synthesises {NUM} peer-reviewed articles published between 2018 and 2024.",
        "Research objectives were aligned with the UN Sustainable Development Goals {NUM} and {NUM}.",
        "Methodology: mixed-methods design combining semi-structured interviews with survey data (n={NUM}).",
        "The hypothesis that X causes Y was supported at the 95% confidence level (OR={NUM}.{NUM}, CI [{NUM},{NUM}]).",
        "Grant funding of ${NUM},000 was awarded by the National Science Foundation for this project.",
        "Data collection instruments were validated for reliability (Cronbach's α = 0.{NUM}).",
        "The systematic review followed PRISMA guidelines with {NUM} studies meeting inclusion criteria.",
        "Findings suggest a statistically significant correlation (r={NUM}.{NUM}, p={NUM}.0{NUM}).",
        "Ethical approval was obtained from the institutional review board prior to data collection.",
        "Limitations include selection bias and a sample size that may limit generalisability.",
        "Future research should explore longitudinal effects across {NUM}+ year time horizons.",
        "The model explains {NUM}% of variance in the dependent variable (R²={NUM}.{NUM}).",
        "Replication of these findings across diverse populations is recommended to confirm validity.",
    ],
    "Compliance": [
        "The audit identified {NUM} control deficiencies, of which {NUM} are classified as material weaknesses.",
        "GDPR Article {NUM} compliance requires a lawful basis for processing personal data of EU residents.",
        "The anti-money laundering policy mandates enhanced due diligence for transactions exceeding ${NUM},000.",
        "ISO {NUM}:{NUM} certification was renewed following a successful third-party surveillance audit.",
        "Regulatory filing was submitted to the SEC on {DATE} in accordance with Form 10-K requirements.",
        "The whistleblower programme received {NUM} reports this quarter, all investigated within {NUM} days.",
        "Sanctions screening must be completed before onboarding any new counterparty or supplier.",
        "The vendor risk assessment framework rates {NUM}% of suppliers as high-risk requiring annual review.",
        "Data retention schedules have been updated to comply with the {NUM}-year statutory requirement.",
        "Penetration testing conducted quarterly identified {NUM} critical vulnerabilities, all now remediated.",
        "The board received a compliance dashboard report highlighting {NUM} open action items.",
        "Training completion rates for mandatory compliance modules stand at {NUM}% across the organisation.",
        "An independent review confirmed adherence to SOX Section {NUM} internal control requirements.",
        "The privacy impact assessment was completed before launching the new customer data platform.",
        "Non-compliance penalties under the regulation can reach up to {NUM}% of annual global turnover.",
    ],
}

DATES = ["January 15, 2024", "March 3, 2024", "June 30, 2024", "September 1, 2023", "December 31, 2023"]


def _fill(template: str) -> str:
    text = template
    text = text.replace("{NUM}", str(random.randint(1, 999)))
    text = text.replace("{DATE}", random.choice(DATES))
    return text


def _generate_document(category: str) -> str:
    templates = TEMPLATES[category]
    n_sentences = random.randint(4, 9)
    chosen = random.choices(templates, k=n_sentences)
    return " ".join(_fill(t) for t in chosen)


def generate_dataset(n: int = 2600, seed: int = 42) -> list[dict]:
    random.seed(seed)
    categories = list(TEMPLATES.keys())
    records = []
    per_cat = n // len(categories)
    remainder = n % len(categories)

    for i, cat in enumerate(categories):
        count = per_cat + (1 if i < remainder else 0)
        for _ in range(count):
            records.append({
                "document_id": str(uuid.uuid4()),
                "category": cat,
                "text": _generate_document(cat),
            })

    random.shuffle(records)
    return records


def save_csv(records: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "category", "text"])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic document corpus")
    parser.add_argument("--n", type=int, default=2600, help="Number of documents")
    parser.add_argument("--output", default="data/raw/documents.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.n} documents …")
    records = generate_dataset(n=args.n, seed=args.seed)
    save_csv(records, args.output)

    cat_counts = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    print(f"\nDataset saved to: {args.output}")
    print(f"Total records:    {len(records)}")
    print("\nCategory distribution:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<15} {cnt:>5} documents")


if __name__ == "__main__":
    main()
