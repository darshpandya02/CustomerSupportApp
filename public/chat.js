(() => {
  const $ = (id) => document.getElementById(id);
  const log = $("log");
  let conversationId = sessionStorage.getItem("conversation_id");
  let lastQuestion = "";
  let ticketKey = null;

  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));

  function renderText(text) {
    // Escape, then turn [n] markers into superscript citation links.
    return esc(text).replace(/\[(\d+)\]/g, '<a class="cite" href="#src-$1">[$1]</a>').replace(/\n/g, "<br>");
  }

  function add(role, html, cls = "") {
    const div = document.createElement("div");
    div.className = `msg ${role} ${cls}`;
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function renderAnswer(j) {
    const badge = { llm: "Answer", retrieval_only: "Manual sections", refusal: "Not in the manual" }[j.mode];
    let html = `<div class="badge ${j.mode}">${badge}</div>`;
    if (j.mode === "retrieval_only") {
      html += `<p class="notice">${esc(j.notice || "")}</p>`;
      html += j.citations.map((c) => `
        <div class="passage" id="src-${c.n}">
          <div class="p-head">[${c.n}] <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)} — ${esc(c.section)}</a></div>
          <div class="p-body">${esc(c.snippet)}</div>
        </div>`).join("");
    } else {
      html += `<div class="text">${renderText(j.answer)}</div>`;
      if (j.citations.length) {
        html += `<ol class="sources">` + j.citations.map((c) =>
          `<li id="src-${c.n}" value="${c.n}"><a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)} — ${esc(c.section)}</a></li>`).join("") + `</ol>`;
      }
    }
    if (j.mode === "refusal") html += `<button type="button" class="secondary small" data-escalate>Escalate to a human</button>`;
    return html;
  }

  async function send(message) {
    add("user", esc(message));
    const pending = add("bot", '<span class="typing">Searching the manual…</span>');
    $("send").disabled = true;
    const key = uid();
    const body = { message };
    if (conversationId) body.conversation_id = conversationId;
    let res;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        // Same Idempotency-Key on retry: the server replays instead of answering twice.
        res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key }, body: JSON.stringify(body) });
        break;
      } catch (e) { if (attempt) { res = null; } }
    }
    $("send").disabled = false;
    if (!res) { pending.innerHTML = "Network error. Please try again."; return; }
    const j = await res.json().catch(() => ({}));
    if (res.status === 429) { pending.innerHTML = `You're sending messages too quickly. Try again in ${j.retry_after_s || 60} s.`; return; }
    if (!res.ok) { pending.innerHTML = esc(typeof j.detail === "string" ? j.detail : "Something went wrong."); return; }
    conversationId = j.conversation_id;
    sessionStorage.setItem("conversation_id", conversationId);
    pending.innerHTML = renderAnswer(j);
    pending.dataset.mode = j.mode;
  }

  $("ask").addEventListener("submit", (e) => {
    e.preventDefault();
    const m = $("msg").value.trim();
    if (!m) return;
    lastQuestion = m;
    $("msg").value = "";
    send(m);
  });
  $("msg").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); }
  });

  function openTicketForm() {
    $("ticket-form").hidden = false;
    $("ticket-result").hidden = true;
    $("t-desc").value = $("t-desc").value || lastQuestion;
    ticketKey = ticketKey || uid();
    $("t-desc").focus();
  }
  $("escalate").addEventListener("click", openTicketForm);
  log.addEventListener("click", (e) => { if (e.target.matches("[data-escalate]")) openTicketForm(); });
  $("t-cancel").addEventListener("click", () => { $("ticket-form").hidden = true; });

  $("ticket-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { description: $("t-desc").value.trim() };
    const email = $("t-email").value.trim();
    if (email) body.email = email;
    if (conversationId) body.conversation_id = conversationId;
    const res = await fetch("/api/tickets", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": ticketKey }, body: JSON.stringify(body) });
    const j = await res.json().catch(() => ({}));
    const out = $("ticket-result");
    out.hidden = false;
    if (!res.ok) {
      const msg = Array.isArray(j.detail) ? j.detail.map((d) => d.msg).join("; ") : (j.detail || "Could not create the ticket.");
      out.innerHTML = `<p class="error">${esc(msg)}</p>`;
      if (res.status !== 422) ticketKey = null;
      return;
    }
    $("ticket-form").hidden = true;
    ticketKey = null;
    const track = `/api/tickets/${encodeURIComponent(j.ref)}?token=${encodeURIComponent(j.tracking_token)}`;
    out.innerHTML = `<p><strong>Ticket ${esc(j.ref)} created.</strong> Routed to the <strong>${esc(j.queue)}</strong> queue
      with <strong>${esc(j.priority)}</strong> priority (${esc(j.routing.method)}). First response due by
      ${new Date(j.first_response_due).toLocaleString()}.</p>
      <p class="muted">${j.transcript_messages} chat messages attached. <a href="${track}" target="_blank" rel="noopener">Track this ticket</a></p>`;
  });

  $("new-chat").addEventListener("click", () => {
    conversationId = null;
    sessionStorage.removeItem("conversation_id");
    log.innerHTML = "";
    $("ticket-result").hidden = true;
    greet();
  });

  function greet() {
    add("bot", "Hi! Ask me anything about using Nextcloud. I answer from the official user manual and link the sections I used.");
  }
  greet();

  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("status").textContent = h.llm && h.llm.available === false
      ? "LLM answers are temporarily unavailable, so the assistant is showing the most relevant manual sections."
      : "";
  }).catch(() => {});
})();
