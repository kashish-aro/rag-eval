# Atlas Rate Limits and Quotas

## Request Rate Limits

Rate limits are enforced per API key, not per account. An account with several keys receives the limit on each key independently.

Starter keys are limited to 100 requests per minute. Growth keys are limited to 1,000 requests per minute. Enterprise keys are limited to 5,000 requests per minute.

## Burst

All tiers receive a burst allowance of twice the standard limit for a window of 10 seconds. Sustained traffic above the standard limit after the burst window is rejected.

## Rejection Behavior

Requests over the limit are rejected with HTTP 429 and a Retry-After header giving the number of seconds until capacity is available. Clients are expected to honor Retry-After. Clients that retry without honoring it may have the key throttled for one hour.

## Execution Quotas

Workflow executions are quota-limited separately from API requests. Starter accounts are limited to 10,000 executions per month, Growth to 250,000, and Enterprise is unmetered. Quota usage resets on the first of the month.

Accounts at 90 percent of monthly execution quota receive an email to the account owner. Executions beyond quota are queued rather than dropped and run when quota resets.
