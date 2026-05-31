# Session 7 Sanity Check

Executed with `text-embedding-3-small` through `scripts/compare.py`.

| Pair | Text A | Text B | Cosine similarity |
| --- | --- | --- | --- |
| A | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | 0.5957 |
| B | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | 0.1920 |
| C | Backend services | API development | 0.5407 |

Pair A lands just below the orientative 0.6 threshold, but still reads as a close
match: both texts describe authentication/authorization with JWTs in a financial
application context. Pair B is much lower, which matches the intuition that
identity flows and database migrations should not cluster together. Pair C lands
in the middle because both phrases are generic backend/software delivery language;
it is related, but not as specific as the fintech authentication pair.
