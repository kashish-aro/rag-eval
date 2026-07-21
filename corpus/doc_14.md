# Atlas API Authentication

## API Keys

The Atlas API authenticates with API keys passed as a bearer token in the Authorization header. Keys are created in the Atlas console under Settings, API Keys.

Keys are shown exactly once at creation and cannot be retrieved afterward. A lost key must be revoked and replaced.

## Scopes

Every key is issued with one of three scopes: read, write, or admin. Read keys may list and inspect workflows and executions. Write keys may additionally create, update, and trigger workflows. Admin keys may additionally manage other keys and account settings.

Scopes cannot be changed after a key is created.

## Rotation

Keys must be rotated every 90 days. Atlas emails the account owner 14 days before expiry and again at 3 days. Expired keys return HTTP 401 with an error code of key_expired.

Revoking a key takes effect immediately across all regions.

## OAuth

Third-party applications integrating with Atlas use OAuth 2.0 authorization code flow rather than API keys. OAuth tokens expire after one hour and are refreshed with a refresh token valid for 30 days.

## Errors

Authentication failures return HTTP 401. Authorization failures, meaning a valid key with insufficient scope, return HTTP 403 with an error code of insufficient_scope.
