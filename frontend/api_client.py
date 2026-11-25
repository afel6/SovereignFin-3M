"""HTTP access for the dashboard, using the shared request and response models."""

from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

from orchestrator.api_contract import AskRequest, AskResponse, ErrorDetail, HealthResponse, Mode

INVALID_QUESTION = "Enter a question of 1 to 2,000 characters."


@dataclass(frozen=True)
class ApiError:
    message: str
    status_code: int | None = None
    transport_error: bool = False


def _request[Response: BaseModel](
    client: httpx.Client,
    method: str,
    path: str,
    model: type[Response],
    **kwargs,
) -> Response | ApiError:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        return ApiError("The API request timed out.", transport_error=True)
    except (httpx.RequestError, httpx.InvalidURL):
        return ApiError("Cannot reach the API.", transport_error=True)

    status = response.status_code
    if status == 422:
        return ApiError("The API rejected the request.", status)
    if status == 503:
        try:
            return ApiError(ErrorDetail.model_validate_json(response.content).detail, status)
        except ValidationError:
            return ApiError("The API is unavailable.", status)
    if status != 200:
        return ApiError("The API returned an unexpected HTTP response.", status)
    try:
        return model.model_validate_json(response.content)
    except ValidationError:
        return ApiError("The API returned an invalid response.", status)


def ask(
    client: httpx.Client, question: str, mode: Mode, thread_id: str | None = None
) -> AskResponse | ApiError:
    try:
        request = AskRequest(question=question, mode=mode, thread_id=thread_id)
    except ValidationError as error:
        if any(item["loc"] == ("question",) for item in error.errors()):
            return ApiError(INVALID_QUESTION)
        return ApiError("The question settings are invalid.")
    return _request(client, "POST", "/ask", AskResponse, json=request.model_dump(mode="json"))


def health(client: httpx.Client) -> HealthResponse | ApiError:
    return _request(client, "GET", "/health", HealthResponse)
