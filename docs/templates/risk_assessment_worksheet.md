# Model Risk Assessment Worksheet

**Model Name:** _______________________________
**Assessment Date:** _______________________________
**Assessor:** _______________________________

---

## Instructions

This worksheet helps you determine the appropriate risk tier for your AI/ML model. Answer each question and use the scoring guide to calculate your model's risk tier.

**Risk Tiers:**
- **Low:** Minimal impact, limited scope, well-understood technology
- **Medium:** Moderate impact, broader scope, some uncertainty
- **High:** Significant impact, wide scope, material consequences
- **Critical:** Mission-critical, regulatory impact, severe consequences

---

## Section 1: Business Impact Assessment

### 1.1 What decisions does this model influence?

- [ ] **Informational only** (analytics, reporting) → +0 points
- [ ] **Decision support** (recommendations that humans review) → +1 point
- [ ] **Automated decisions** (decisions made with minimal human oversight) → +2 points
- [ ] **Fully automated critical decisions** (no human in the loop) → +3 points

**Score:** _____ / 3

### 1.2 What is the financial impact if the model makes errors?

- [ ] **Minimal** (<$10K potential loss) → +0 points
- [ ] **Low** ($10K - $100K) → +1 point
- [ ] **Medium** ($100K - $1M) → +2 points
- [ ] **High** ($1M - $10M) → +3 points
- [ ] **Severe** (>$10M) → +4 points

**Score:** _____ / 4

### 1.3 How many customers/users are affected by this model?

- [ ] **Limited** (<100) → +0 points
- [ ] **Moderate** (100 - 1,000) → +1 point
- [ ] **Significant** (1,000 - 10,000) → +2 points
- [ ] **Large** (10,000 - 100,000) → +3 points
- [ ] **Very large** (>100,000) → +4 points

**Score:** _____ / 4

### 1.4 What is the potential reputational impact?

- [ ] **None** (internal tool, no external visibility) → +0 points
- [ ] **Minor** (limited customer complaints possible) → +1 point
- [ ] **Moderate** (negative PR possible) → +2 points
- [ ] **Major** (significant brand damage possible) → +3 points
- [ ] **Severe** (existential threat to organization) → +4 points

**Score:** _____ / 4

**Section 1 Total:** _____ / 15

---

## Section 2: Regulatory & Compliance Impact

### 2.1 Does this model fall under regulatory oversight?

- [ ] **No regulatory requirements** → +0 points
- [ ] **Industry guidelines** (non-binding) → +1 point
- [ ] **Regulatory guidance** (SR 11-7, GDPR, etc.) → +2 points
- [ ] **Direct regulatory requirement** (must comply) → +3 points
- [ ] **High-risk regulatory application** (banking, healthcare) → +4 points

**Score:** _____ / 4

### 2.2 What is the potential regulatory penalty for non-compliance?

- [ ] **Not applicable** → +0 points
- [ ] **Warning/notice** → +1 point
- [ ] **Fines <$100K** → +2 points
- [ ] **Fines $100K - $1M** → +3 points
- [ ] **Fines >$1M or license revocation** → +4 points

**Score:** _____ / 4

### 2.3 Does the model involve protected attributes or sensitive data?

- [ ] **No sensitive data** → +0 points
- [ ] **Business data only** → +1 point
- [ ] **Personal data (non-protected)** → +2 points
- [ ] **Protected attributes** (race, gender, age, etc.) → +3 points
- [ ] **Special category data** (health, biometric, financial) → +4 points

**Score:** _____ / 4

**Section 2 Total:** _____ / 12

---

## Section 3: Model Complexity & Uncertainty

### 3.1 What is the model's complexity level?

- [ ] **Simple** (linear regression, rule-based) → +0 points
- [ ] **Moderate** (decision trees, logistic regression) → +1 point
- [ ] **Complex** (random forest, gradient boosting) → +2 points
- [ ] **Highly complex** (deep neural networks) → +3 points
- [ ] **Cutting-edge** (experimental architectures, LLMs) → +4 points

**Score:** _____ / 4

### 3.2 How interpretable is the model?

- [ ] **Fully interpretable** (can explain every decision) → +0 points
- [ ] **Mostly interpretable** (can explain most decisions) → +1 point
- [ ] **Partially interpretable** (can explain general behavior) → +2 points
- [ ] **Limited interpretability** (black box with some insights) → +3 points
- [ ] **Black box** (no meaningful interpretation possible) → +4 points

**Score:** _____ / 4

### 3.3 How mature is the technology used?

- [ ] **Well-established** (>10 years in production use) → +0 points
- [ ] **Proven** (5-10 years, widely adopted) → +1 point
- [ ] **Emerging** (2-5 years, growing adoption) → +2 points
- [ ] **New** (<2 years, limited production use) → +3 points
- [ ] **Experimental** (research stage, unproven) → +4 points

**Score:** _____ / 4

### 3.4 What is the model's performance certainty?

- [ ] **High certainty** (>95% accuracy, well-validated) → +0 points
- [ ] **Good certainty** (85-95% accuracy) → +1 point
- [ ] **Moderate certainty** (75-85% accuracy) → +2 points
- [ ] **Lower certainty** (65-75% accuracy) → +3 points
- [ ] **Uncertain** (<65% accuracy or unknown) → +4 points

**Score:** _____ / 4

**Section 3 Total:** _____ / 16

---

## Section 4: Data & Operational Risk

### 4.1 What is the quality of training data?

- [ ] **Excellent** (clean, complete, representative) → +0 points
- [ ] **Good** (minor quality issues) → +1 point
- [ ] **Fair** (some quality concerns) → +2 points
- [ ] **Poor** (significant quality issues) → +3 points
- [ ] **Unknown** (data quality not assessed) → +4 points

**Score:** _____ / 4

### 4.2 How likely is data drift?

- [ ] **Very stable** (static data, no expected changes) → +0 points
- [ ] **Stable** (slow-changing, predictable) → +1 point
- [ ] **Moderate** (some drift expected) → +2 points
- [ ] **Dynamic** (frequent drift likely) → +3 points
- [ ] **Highly dynamic** (rapid, unpredictable drift) → +4 points

**Score:** _____ / 4

### 4.3 What is the operational deployment complexity?

- [ ] **Simple** (batch processing, offline) → +0 points
- [ ] **Moderate** (API calls, low volume) → +1 point
- [ ] **Complex** (real-time, moderate volume) → +2 points
- [ ] **Highly complex** (real-time, high volume, SLA) → +3 points
- [ ] **Mission-critical** (24/7, zero downtime required) → +4 points

**Score:** _____ / 4

### 4.4 How reversible are model decisions?

- [ ] **Fully reversible** (can easily undo decisions) → +0 points
- [ ] **Mostly reversible** (can reverse with some effort) → +1 point
- [ ] **Partially reversible** (some decisions cannot be undone) → +2 points
- [ ] **Difficult to reverse** (significant cost/effort to undo) → +3 points
- [ ] **Irreversible** (decisions cannot be undone) → +4 points

**Score:** _____ / 4

**Section 4 Total:** _____ / 16

---

## Section 5: Fairness & Ethical Considerations

### 5.1 Could the model create disparate impact?

- [ ] **No impact on individuals** → +0 points
- [ ] **Equal impact across groups** → +1 point
- [ ] **Potential for minor disparities** → +2 points
- [ ] **Likely disparate impact** → +3 points
- [ ] **High risk of discrimination** → +4 points

**Score:** _____ / 4

### 5.2 How severe would bias/unfairness be if present?

- [ ] **Not applicable** → +0 points
- [ ] **Minor inconvenience** → +1 point
- [ ] **Moderate harm** (financial, opportunity) → +2 points
- [ ] **Significant harm** (denial of services, rights) → +3 points
- [ ] **Severe harm** (safety, livelihood, dignity) → +4 points

**Score:** _____ / 4

### 5.3 Is there human oversight of model decisions?

- [ ] **Always reviewed by humans** → +0 points
- [ ] **Frequently reviewed** → +1 point
- [ ] **Occasionally reviewed** → +2 points
- [ ] **Rarely reviewed** → +3 points
- [ ] **No human oversight** → +4 points

**Score:** _____ / 4

**Section 5 Total:** _____ / 12

---

## Risk Score Calculation

| Section | Your Score | Maximum |
|---------|------------|---------|
| 1. Business Impact | _____ | 15 |
| 2. Regulatory & Compliance | _____ | 12 |
| 3. Model Complexity | _____ | 16 |
| 4. Data & Operational | _____ | 16 |
| 5. Fairness & Ethics | _____ | 12 |
| **TOTAL** | **_____** | **71** |

**Total Score:** _____ / 71
**Percentage:** _____ %

---

## Risk Tier Determination

Based on your total score, determine the risk tier:

### Option 1: Percentage-based Tiers

- **Low Risk:** 0-20% (0-14 points)
- **Medium Risk:** 21-40% (15-28 points)
- **High Risk:** 41-70% (29-50 points)
- **Critical Risk:** 71-100% (51-71 points)

### Option 2: Factor-based Override

**Automatic Critical Tier if ANY of these are true:**
- [ ] Section 2 (Regulatory) score > 10
- [ ] Section 1.2 (Financial impact) = Severe (>$10M)
- [ ] Section 1.1 (Decision type) = Fully automated critical decisions
- [ ] Section 5.2 (Bias severity) = Severe harm

**Automatic High Tier if ANY of these are true:**
- [ ] Section 2 (Regulatory) score > 7
- [ ] Section 1.2 (Financial impact) = High ($1M-$10M)
- [ ] Section 1.3 (Users affected) > 10,000
- [ ] Section 5.1 (Disparate impact) = Likely or High risk

**Use the higher of the two risk tier determinations.**

---

## Final Risk Tier Assignment

**Calculated Tier (Option 1):** _______________

**Override Tier (Option 2):** _______________

**Final Risk Tier:** ☐ Low  ☐ Medium  ☐ High  ☐ Critical

**Justification:**
[Explain your risk tier assignment, noting any overrides or special considerations]

---

## Governance Requirements by Risk Tier

Based on your final risk tier, the following governance requirements apply:

### Low Risk
- ✅ Basic model card
- ✅ Code review
- ✅ Annual validation
- ✅ Technical lead approval

### Medium Risk
- ✅ Comprehensive model card
- ✅ Technical documentation
- ✅ Semi-annual validation
- ✅ Bias testing
- ✅ Technical lead + validator approval

### High Risk
- ✅ Full documentation suite (model card, technical docs, validation report)
- ✅ Quarterly validation
- ✅ Comprehensive bias and fairness testing
- ✅ Data drift monitoring
- ✅ Multi-level approval (validator + compliance + business owner)
- ✅ Incident response plan

### Critical Risk
- ✅ All high-risk requirements PLUS:
- ✅ Monthly validation
- ✅ Executive-level approval (CRO + Compliance + Validation)
- ✅ Enhanced monitoring and alerting
- ✅ Regular audit trail reviews
- ✅ Detailed incident response and business continuity plans

---

## Decision Matrix Visual Guide

```
                    ┌─────────────────────────────────────────┐
                    │   MODEL RISK CLASSIFICATION GUIDE      │
                    └─────────────────────────────────────────┘

FINANCIAL DECISIONS?
    ├─ YES, >$10M impact ────────────────────────────► CRITICAL
    ├─ YES, $1M-$10M impact ─────────────────────────► HIGH
    ├─ YES, $100K-$1M impact ────────────────────────► MEDIUM/HIGH
    └─ YES, <$100K impact ───────────────────────────► MEDIUM

REGULATORY OVERSIGHT?
    ├─ High-risk regulatory (SR 11-7 critical) ──────► CRITICAL
    ├─ Direct regulatory requirement ────────────────► HIGH
    ├─ Regulatory guidance applies ──────────────────► MEDIUM/HIGH
    └─ No regulatory requirements ───────────────────► No change

CUSTOMER-FACING?
    ├─ Automated decisions, >100K users ─────────────► HIGH/CRITICAL
    ├─ Automated decisions, >10K users ──────────────► HIGH
    ├─ Decision support, >10K users ─────────────────► MEDIUM
    └─ Internal analytics only ──────────────────────► LOW/MEDIUM

PROTECTED ATTRIBUTES?
    ├─ Special category data + high impact ──────────► CRITICAL
    ├─ Protected attributes used ────────────────────► HIGH
    ├─ Personal data only ───────────────────────────► MEDIUM
    └─ No sensitive data ────────────────────────────► No change

BLACK BOX MODEL?
    ├─ Black box + high stakes ──────────────────────► Increase tier
    ├─ Complex but interpretable ────────────────────► No change
    └─ Fully interpretable ──────────────────────────► No change
```

---

## Next Steps

After determining your model's risk tier:

1. **Update Model Registry**
   - Add or update entry in `inventory/model_registry.yaml`
   - Include risk tier and justification

2. **Prepare Required Documentation**
   - Use templates in `docs/templates/` based on your risk tier
   - Ensure all required documentation is complete

3. **Set Up Monitoring**
   - Configure monitoring based on risk tier requirements
   - Set up alerts and dashboards

4. **Plan Validation Schedule**
   - Schedule validations according to risk tier:
     - Critical: Monthly
     - High: Quarterly
     - Medium: Semi-annual
     - Low: Annual

5. **Identify Approvers**
   - Determine who needs to approve based on risk tier
   - Ensure approvers are documented in registry

---

## Assessor Sign-off

**Assessor Name:** _______________________________
**Title:** _______________________________
**Date:** _______________________________
**Signature:** _______________________________

**Reviewer Name:** _______________________________ *(if required)*
**Title:** _______________________________
**Date:** _______________________________
**Signature:** _______________________________

---

## Appendix: Example Scenarios

### Example 1: Low Risk Model
**Model:** Internal dashboard showing sales trends
- Informational only (+0)
- No financial decisions (<$10K) (+0)
- Internal only (<100 users) (+0)
- No regulatory requirements (+0)
- Simple aggregation (+0)
**Total Score:** 2/71 → **Low Risk**

### Example 2: Medium Risk Model
**Model:** Product recommendation engine
- Decision support (+1)
- Low financial impact (+1)
- 5,000 users (+2)
- Personal data (+2)
- Moderate complexity (+1)
**Total Score:** 18/71 → **Medium Risk**

### Example 3: High Risk Model
**Model:** Credit approval model
- Automated decisions (+2)
- Medium financial impact ($500K) (+2)
- 50,000 customers (+3)
- Regulatory guidance (SR 11-7) (+2)
- Protected attributes (+3)
- Complex model (+2)
**Total Score:** 38/71 → **High Risk**

### Example 4: Critical Risk Model
**Model:** Fraud detection for payment processing
- Fully automated (+3)
- Severe financial impact (>$10M) (+4)
- >1M users (+4)
- Direct regulatory requirement (+3)
- Special category financial data (+4)
- Real-time, mission-critical (+4)
**Total Score:** 58/71 → **Critical Risk**

---

**Worksheet Version:** 1.0
**Last Updated:** 2026-01-16
