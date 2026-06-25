from __future__ import annotations

import pytest


def test_favicon_returns_no_content() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 204


def test_personal_page_includes_visible_workflow_loading_state() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="workflow-loading"' in html
    assert 'role="status"' in html
    assert "사주 리포트 생성 중입니다" in html
    assert "loading.hidden = false" in html
