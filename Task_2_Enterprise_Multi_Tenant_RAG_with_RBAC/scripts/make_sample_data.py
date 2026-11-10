"""Deterministic generator for the 8-document multi-format enterprise sample corpus."""
import csv
import json
from pathlib import Path
import fitz
import yaml

from src.config import BASE_DIR, DEFAULT_TENANT


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 780),
            page_text,
            fontsize=11,
            fontname="helv",
        )
    doc.save(str(path))
    doc.close()


def generate_sample_corpus(out_dir: Path | None = None) -> Path:
    """
    Generate the 8 enterprise sample documents and manifest.yaml in out_dir.
    Uses pure Python strings so all currency symbols ($184,500,000) and backticks
    (`vpc-prod-compute`) are preserved verbatim.
    """
    target_dir = Path(out_dir) if out_dir is not None else (BASE_DIR / "data" / "sample_documents")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. security_compliance_tier1.pdf (operations, Tier 1, 3 pages)
    _write_pdf(
        target_dir / "security_compliance_tier1.pdf",
        [
            (
                "Enterprise Security & Compliance Policy (Tier 1 - Public Internal)\n\n"
                "Section 1: Acceptable Use and Device Security\n"
                "All corporate workstations must run full-disk encryption (BitLocker or FileVault) "
                "and maintain active endpoint telemetry via the CrowdStrike Falcon agent. Employees must "
                "authenticate using hardware-backed FIDO2 multi-factor authentication (MFA) keys across "
                "all single sign-on (SSO) portals. Personal mobile devices accessing corporate email must "
                "enroll in the Mobile Device Management (MDM) container profile."
            ),
            (
                "Section 2: Data Classification and Password Hygiene\n"
                "Corporate information is categorized into four strict clearance tiers: Tier 1 (Public Internal), "
                "Tier 2 (Departmental Operational), Tier 3 (Leadership & Financial Confidential), and Tier 4 "
                "(Executive & Legal Restricted). Standard user passwords must be at least 16 characters in length "
                "and rotated immediately upon any suspected compromise. Session idle lockout is enforced at 15 minutes."
            ),
            (
                "Section 3: Incident Reporting and Security Operations Center (SOC)\n"
                "Any suspected phishing attempt, credential exposure, or anomalous network egress must be "
                "reported to the Security Operations Center (SOC) within 15 minutes of discovery via "
                "soc-alerts@enterprise.internal or extension 4911. The SOC maintains 24/7/365 triage coverage."
            ),
        ],
    )

    # 2. merger_acquisition_audit_tier4.pdf (legal, Tier 4, 2 pages)
    _write_pdf(
        target_dir / "merger_acquisition_audit_tier4.pdf",
        [
            (
                "STRICTLY CONFIDENTIAL - TIER 4 LEGAL & EXECUTIVE ONLY\n"
                "Project Titan: Merger & Acquisition Due Diligence Audit\n\n"
                "1. Deal Valuation and Structure\n"
                "Acquirer (Enterprise Holdings Inc.) proposes to acquire 100% of the issued and outstanding "
                "equity of Target Alpha Dynamics Inc. under Project Titan for an aggregate purchase price of "
                "$184,500,000 ($184.5M), comprising $110,700,000 in cash and $73,800,000 in restricted common stock. "
                "Founder buyout allocation is fixed at $92,000,000 with a 36-month cliff vesting schedule. "
                "Employee retention pool is established at $25,000,000. Intellectual property audit verified "
                "14 granted US patents with zero pending litigation."
            ),
            (
                "2. Escrow, Indemnification, and Key Covenants\n"
                "An indemnity escrow account of $18,450,000 (10% of aggregate deal consideration) shall be held "
                "by J.P. Morgan Escrow Services for a period of 18 months post-closing to cover breaches of "
                "representations, tax liabilities, and undisclosed environmental claims. Closing is contingent "
                "upon Target Alpha Dynamics maintaining minimum audited annual recurring revenue (ARR) of "
                "$22,000,000 and key executive principals executing 4-year non-compete agreements."
            ),
        ],
    )

    # 3. hr_compensation_bands_tier2.csv (hr, Tier 2)
    with open(target_dir / "hr_compensation_bands_tier2.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "role_title", "department", "base_min_usd", "base_midpoint_usd", "base_max_usd", "target_bonus_pct", "equity_refresh_usd"])
        writer.writerow(["L3", "Associate Software Engineer", "Engineering", "115000", "130000", "145000", "10%", "25000"])
        writer.writerow(["L4", "Senior Software Engineer", "Engineering", "150000", "168000", "185000", "15%", "55000"])
        writer.writerow(["L5", "Staff Software Engineer", "Engineering", "185000", "210000", "240000", "20%", "95000"])
        writer.writerow(["L6", "Principal Architect", "Engineering", "235000", "265000", "295000", "25%", "160000"])
        writer.writerow(["M3", "Engineering Manager", "Engineering", "190000", "215000", "245000", "20%", "100000"])

    # 4. finance_q3_budget_audit_tier3.csv (finance, Tier 3)
    with open(target_dir / "finance_q3_budget_audit_tier3.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cost_center_id", "department", "category", "q3_allocated_usd", "q3_actual_usd", "variance_usd", "audit_status", "auditor_notes"])
        writer.writerow(["FIN-Q3-001", "Engineering", "Cloud Compute & GPU Clusters", "1250000", "1390000", "-140000", "Flagged", "Overrun driven by H100 training burst in us-east-1"])
        writer.writerow(["FIN-Q3-002", "Operations", "Enterprise SaaS & Snowflake", "450000", "480000", "-30000", "Reviewed", "Annual seat true-up and Snowflake warehouse scaling"])
        writer.writerow(["FIN-Q3-003", "Legal", "External Patent & M&A Counsel", "620000", "595000", "25000", "Approved", "Under budget following early settlement of vendor review"])
        writer.writerow(["FIN-Q3-004", "Executive", "Strategic Reserve & Advisory", "800000", "800000", "0", "Approved", "Retainer disbursements matched forecast"])

    # 5. engineering_cloud_vpc_runbook_tier2.md (engineering, Tier 2)
    (target_dir / "engineering_cloud_vpc_runbook_tier2.md").write_text(
        "# Cloud Infrastructure & VPC Peering Runbook\n\n"
        "## 1. Network Topology & Peering\n"
        "Production workloads run inside the primary VPC (`vpc-prod-compute`) with CIDR block `10.42.0.0/16` "
        "in `us-east-1`, peered with the shared services VPC (`vpc-shared-sec`) on `10.10.0.0/16`. All cross-VPC "
        "traffic must traverse AWS Transit Gateway `tgw-09a8b7c6d5e4f3210` with strict Network ACL enforcement.\n\n"
        "## 2. Routing, NAT Egress & Failover\n"
        "Outbound internet traffic from private subnets (`10.42.16.0/20` and `10.42.32.0/20`) routes through "
        "redundant NAT Gateways deployed across three Availability Zones (`us-east-1a`, `us-east-1b`, `us-east-1c`). "
        "BGP dynamic routing over AWS Direct Connect uses ASN `64512` with bidirectional forwarding detection (BFD) "
        "configured at 300ms intervals for sub-second failover.\n",
        encoding="utf-8",
    )

    # 6. executive_ai_roadmap_tier3.md (executive, Tier 3)
    (target_dir / "executive_ai_roadmap_tier3.md").write_text(
        "# Executive AI Strategy & FY2026 Capital Roadmap\n\n"
        "## 1. Capital Allocation & Infrastructure Budget\n"
        "The Board of Directors has approved a total FY2026 AI capital expenditure (CapEx) budget of $14.5M "
        "($14,500,000) dedicated to sovereign LLM inference clusters, multi-tenant vector infrastructure, and "
        "autonomous field engineering deployment. Phase 1 GPU cluster reservation accounts for $8.1M.\n\n"
        "## 2. Projected Cost Savings & Margin Expansion\n"
        "Full deployment of the Enterprise Multi-Tenant RAG platform and Autonomous Field Agent across Tier-2 "
        "customer support and legacy migration workflows is projected to deliver $7.2M ($7,200,000) in annualized "
        "operating expense reductions and improve gross EBITDA margin by 340 basis points by Q4 FY2026.\n",
        encoding="utf-8",
    )

    # 7. operations_supplier_catalog_tier1.json (operations, Tier 1)
    suppliers_payload = {
        "catalog_version": "2026.11",
        "owner_department": "Operations",
        "records": [
            {
                "supplier_id": "SUP-101",
                "vendor_name": "Apex Global Logistics",
                "tier_status": "Preferred",
                "sla_uptime": "99.95%",
                "primary_region": "North America",
                "annual_contract_limit_usd": 1200000,
            },
            {
                "supplier_id": "SUP-102",
                "vendor_name": "Meridian Cloud Hardware",
                "tier_status": "Strategic",
                "sla_uptime": "99.99%",
                "primary_region": "Global",
                "annual_contract_limit_usd": 4500000,
            },
            {
                "supplier_id": "SUP-103",
                "vendor_name": "Vanguard Facility Services",
                "tier_status": "Standard",
                "sla_uptime": "99.50%",
                "primary_region": "EMEA",
                "annual_contract_limit_usd": 650000,
            },
        ],
    }
    (target_dir / "operations_supplier_catalog_tier1.json").write_text(
        json.dumps(suppliers_payload, indent=2),
        encoding="utf-8",
    )

    # 8. legal_ip_license_agreement_tier4.txt (legal, Tier 4)
    (target_dir / "legal_ip_license_agreement_tier4.txt").write_text(
        "CONFIDENTIAL INTELLECTUAL PROPERTY CROSS-LICENSE AGREEMENT (TIER 4 RESTRICTED)\n\n"
        "This Exclusive Patent and Cryptographic Kernel Cross-License Agreement ('Agreement') is entered into "
        "effective October 28, 2026, by and between Enterprise Holdings Inc. ('Licensor') and Quantum Core "
        "Systems Ltd. ('Licensee').\n\n"
        "Section 1. Licensed Patent Portfolio: Licensor grants Licensee a non-transferable, field-restricted "
        "license to integrate the proprietary 'Aegis-X' zero-knowledge proof acceleration kernel covered under "
        "US Patent Nos. 11,402,891 and 11,884,109.\n\n"
        "Section 2. Upfront Fee and Running Royalties: Licensee shall pay an upfront licensing fee of $3,500,000 "
        "within ten (10) business days of execution, plus a running quarterly royalty of 4.5% on net commercial "
        "revenues derived from licensed hardware appliances, subject to an annual minimum royalty floor of $900,000.\n",
        encoding="utf-8",
    )

    # 9. manifest.yaml
    manifest_data = {
        "documents": [
            {
                "file": "security_compliance_tier1.pdf",
                "department": "operations",
                "clearance_tier": 1,
                "document_date": "2026-10-15",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "operations_supplier_catalog_tier1.json",
                "department": "operations",
                "clearance_tier": 1,
                "document_date": "2026-10-18",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "engineering_cloud_vpc_runbook_tier2.md",
                "department": "engineering",
                "clearance_tier": 2,
                "document_date": "2026-10-20",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "hr_compensation_bands_tier2.csv",
                "department": "hr",
                "clearance_tier": 2,
                "document_date": "2026-10-22",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "finance_q3_budget_audit_tier3.csv",
                "department": "finance",
                "clearance_tier": 3,
                "document_date": "2026-10-25",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "executive_ai_roadmap_tier3.md",
                "department": "executive",
                "clearance_tier": 3,
                "document_date": "2026-10-27",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "merger_acquisition_audit_tier4.pdf",
                "department": "legal",
                "clearance_tier": 4,
                "document_date": "2026-10-28",
                "tenant_id": DEFAULT_TENANT,
            },
            {
                "file": "legal_ip_license_agreement_tier4.txt",
                "department": "legal",
                "clearance_tier": 4,
                "document_date": "2026-10-28",
                "tenant_id": DEFAULT_TENANT,
            },
        ]
    }
    with open(target_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    return target_dir


if __name__ == "__main__":
    out = generate_sample_corpus()
    print(f"Generated sample corpus in {out}")
