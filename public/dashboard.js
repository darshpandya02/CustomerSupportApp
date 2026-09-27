(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "");
  let selected = null;

  async function api(path, opts = {}) {
    const res = await fetch(path, { credentials: "same-origin", headers: { "Content-Type": "application/json" }, ...opts });
    if (res.status === 401) { showLogin(); throw new Error("unauthorized"); }
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    return j;
  }

  function showLogin() { $("login").hidden = false; $("app").hidden = true; $("logout").hidden = true; }
  function showApp() { $("login").hidden = true; $("app").hidden = false; $("logout").hidden = false; load(); }

  $("login").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("login-err").textContent = "";
    const res = await fetch("/api/agent/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: $("pw").value }) });
    if (res.ok) { $("pw").value = ""; showApp(); }
    else $("login-err").textContent = res.status === 429 ? "Too many attempts, wait a few minutes." : "Wrong password.";
  });
  $("logout").addEventListener("click", async () => { await fetch("/api/agent/logout", { method: "POST" }); showLogin(); });

  async function load() {
    const p = new URLSearchParams();
    for (const [k, id] of [["queue", "f-queue"], ["status", "f-status"], ["priority", "f-priority"], ["q", "f-q"]]) {
      if ($(id).value) p.set(k, $(id).value);
    }
    const [list, stats] = await Promise.all([api(`/api/agent/tickets?${p}`), api("/api/agent/stats")]);
    const open = {};
    for (const r of stats.by_queue_status) if (!["resolved", "closed"].includes(r.status)) open[r.queue] = (open[r.queue] || 0) + r.n;
    $("stats").innerHTML = ["billing", "technical", "account"].map((q) =>
      `<div class="stat"><div class="n">${open[q] || 0}</div><div class="l">open ${q}</div></div>`).join("") +
      `<div class="stat"><div class="n">${stats.sla_breached_open}</div><div class="l">SLA breached</div></div>`;
    $("rows").innerHTML = list.items.map((t) => `
      <tr data-id="${t.id}" class="${t.id === selected ? "sel" : ""}">
        <td>${esc(t.ref)}</td><td>${esc(t.subject)}</td>
        <td><span class="pill q-${esc(t.queue)}">${esc(t.queue)}</span></td>
        <td><span class="pill p-${esc(t.priority)}">${esc(t.priority)}</span>${t.sla_breached ? ' <span class="pill breach">SLA</span>' : ""}</td>
        <td>${esc(t.status.replaceAll("_", " "))}</td><td>${fmt(t.created_at)}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No tickets match.</td></tr>`;
    if (selected) openTicket(selected);
  }

  $("rows").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (!tr) return;
    selected = Number(tr.dataset.id);
    document.querySelectorAll("#rows tr").forEach((r) => r.classList.toggle("sel", r === tr));
    openTicket(selected);
  });
  for (const id of ["f-queue", "f-status", "f-priority"]) $(id).addEventListener("change", load);
  $("f-q").addEventListener("input", () => { clearTimeout(window.__q); window.__q = setTimeout(load, 300); });
  $("refresh").addEventListener("click", load);

  async function openTicket(id) {
    const t = await api(`/api/agent/tickets/${id}`);
    const r = t.routing || {};
    const scores = r.scores ? Object.entries(r.scores).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(", ") : "";
    const opt = (vals, cur) => vals.map((v) => `<option ${v === cur ? "selected" : ""}>${v}</option>`).join("");
    $("detail").innerHTML = `
      <h3>${esc(t.ref)} · ${esc(t.subject)}</h3>
      <p class="muted">${esc(t.customer_email || "no email")} · created ${fmt(t.created_at)} · first response due ${fmt(t.first_response_due)}</p>
      <p><span class="pill q-${esc(t.queue)}">${esc(t.queue)}</span> <span class="pill p-${esc(t.priority)}">${esc(t.priority)}</span>
        <span class="pill">${esc(t.status.replaceAll("_", " "))}</span></p>
      <p class="routing">Routed by <strong>${esc(r.method)}</strong>${r.rule ? ` (matched “${esc(r.rule)}”)` : ""}${scores ? `; classifier: ${esc(scores)}` : ""}</p>
      <h4>Description</h4><p class="desc">${esc(t.description)}</p>
      <form id="upd" class="upd">
        <label>Status <select name="status"><option value="">${esc(t.status)} (current)</option>${opt(t.allowed_transitions, "")}</select></label>
        <label>Priority <select name="priority">${opt(["urgent", "high", "normal", "low"], t.priority)}</select></label>
        <label>Queue <select name="queue">${opt(["billing", "technical", "account"], t.queue)}</select></label>
        <label>Assignee <input name="assignee" value="${esc(t.assignee || "")}" maxlength="80"></label>
        <label class="wide">Internal note <textarea name="note" rows="2" maxlength="2000"></textarea></label>
        <button type="submit">Save</button> <span id="upd-msg" class="muted"></span>
      </form>
      <h4>Chat transcript (${t.transcript.length})</h4>
      <div class="transcript">${t.transcript.map((m) => `<div class="tmsg ${m.role}"><b>${m.role === "user" ? "Customer" : "Assistant"}${m.mode ? ` · ${esc(m.mode)}` : ""}</b><div>${esc(m.content).replace(/\n/g, "<br>")}</div></div>`).join("") || '<p class="muted">No chat attached.</p>'}</div>
      <h4>History</h4>
      <ul class="events">${t.events.map((e) => `<li>${fmt(e.at)} · <b>${esc(e.actor)}</b> ${esc(e.kind)} ${e.from_value ? esc(e.from_value) + " → " : ""}${esc(e.to_value || "")} ${e.note ? "· " + esc(e.note) : ""}</li>`).join("")}</ul>`;
    $("upd").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const f = new FormData(ev.target);
      const body = {};
      if (f.get("status")) body.status = f.get("status");
      for (const k of ["priority", "queue"]) if (f.get(k) !== t[k]) body[k] = f.get(k);
      if ((f.get("assignee") || "") !== (t.assignee || "")) body.assignee = f.get("assignee");
      if (f.get("note")) body.note = f.get("note");
      try { await api(`/api/agent/tickets/${id}`, { method: "PATCH", body: JSON.stringify(body) }); load(); }
      catch (err) { $("upd-msg").textContent = err.message; }
    });
  }

  fetch("/api/agent/me", { credentials: "same-origin" }).then((r) => (r.ok ? showApp() : showLogin())).catch(showLogin);
})();
