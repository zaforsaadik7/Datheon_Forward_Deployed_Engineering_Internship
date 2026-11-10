"""Authenticated FastAPI service exposing /api/v1/rag/query, /api/v1/rag/personas, and /health."""
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..config import get_settings
from ..etl.vector_store import VectorStore
from ..rag.engine import MultiTenantRAGEngine
from ..rag.models import RAGResponse
from ..rag.rbac import (
    AccessDeniedError,
    AuthenticationError,
    PERSONAS,
    authenticate_api_key,
    authorize_principal,
)


class RAGQueryRequest(BaseModel):
    """
    Strict query payload schema.
    extra='forbid' ensures any attempt to supply clearance_tier, departments, or tenant_id
    in the JSON request body is rejected with HTTP 422.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query must not be blank or whitespace-only.")
        return cleaned


def create_app(
    engine: Optional[MultiTenantRAGEngine] = None,
    allow_dev_persona_header: Optional[bool] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    dev_header_enabled = (
        allow_dev_persona_header
        if allow_dev_persona_header is not None
        else settings.allow_dev_persona_header
    )

    if engine is None:
        store = VectorStore(
            path=settings.chroma_path,
            collection_name=settings.collection_name,
        )
        rag_engine = MultiTenantRAGEngine(store=store)
    else:
        rag_engine = engine

    app = FastAPI(
        title="Datheon Enterprise Multi-Tenant RAG with RBAC",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "collection_count": rag_engine.store.count(),
        }

    @app.get("/api/v1/rag/personas")
    def list_personas() -> Dict[str, List[Dict[str, Any]]]:
        items: List[Dict[str, Any]] = []
        for pid, p in PERSONAS.items():
            items.append(
                {
                    "user_id": pid,
                    "role_title": p.role_title,
                    "tenant_id": p.tenant_id,
                    "clearance_tier": p.clearance_tier,
                    "departments": sorted(p.departments),
                }
            )
        return {"personas": items}

    @app.post("/api/v1/rag/query", response_model=RAGResponse)
    def query_rag(
        payload: RAGQueryRequest,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_dev_persona: Optional[str] = Header(default=None, alias="X-Dev-Persona"),
    ) -> RAGResponse:
        token = x_api_key if x_api_key is not None else authorization
        try:
            principal = authenticate_api_key(
                api_key=token,
                dev_persona=x_dev_persona,
                allow_dev_header=dev_header_enabled,
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        try:
            authorize_principal(principal)
        except AccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        return rag_engine.answer(
            query=payload.query,
            principal=principal,
            top_k=payload.top_k,
        )

    return app
