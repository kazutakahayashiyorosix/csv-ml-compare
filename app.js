/* CSV ML Compare — 画面側の処理。計算は worker.js（Pyodide）に任せる。 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  };

  // ---------------------------------------------------------------- worker
  // 計算エンジンが落ちたら（メモリ不足など）自動で作り直す。待っている処理は失敗として返す。
  let worker = null, catalogue = null, seq = 0, lastText = null, lastName = null;
  const pending = new Map();
  function startWorker() {
    catalogue = null;
    worker = new Worker("worker.js?v=" + APP_VERSION);
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "status") return setEngine(m.text, "");
      if (m.type === "progress") return setRunStatus(m.text, true);
      if (m.type === "ready") {
        catalogue = m.catalogue; setEngine("準備完了", "ok");
        if (lastText && !info) reloadAfterRestart(); else { refreshModels(); updateRunBtn(); }
        return;
      }
      if (m.type === "fatal") { setEngine("計算エンジンの起動に失敗：" + m.text, "ng"); showRetry(); return; }
      const p = pending.get(m.id); if (!p) return; pending.delete(m.id);
      if (m.ok) p.res(m.data); else { const e = new Error(m.error); e.detail = m.detail; p.rej(e); }
    };
    worker.onerror = (e) => { e.preventDefault && e.preventDefault(); crashed(e.message); };
  }
  function crashed(msg) {
    const err = new Error("計算エンジンが止まりました（メモリ不足の可能性）。行や列を減らすか、重いモデル・交差検証を外して、もう一度試してください。");
    err.detail = msg || "";
    for (const p of pending.values()) p.rej(err);
    pending.clear();
    try { worker.terminate(); } catch (_) {}
    setEngine("再起動中…", "");
    info = null;
    startWorker();
  }
  const call = (cmd, args) => new Promise((res, rej) => {
    if (!worker) return rej(new Error("計算エンジンがありません"));
    const id = ++seq; pending.set(id, { res, rej }); worker.postMessage({ id, cmd, args });
  });
  async function reloadAfterRestart() {
    try { info = await call("load", { text: lastText }); refreshModels(); updateRunBtn(); } catch (_) {}
  }
  function showRetry() {
    const b = el("button", { type: "button", class: "link", text: "再読み込み", onclick: () => location.reload() });
    $("engineText").append(" ", b);
  }
  function setEngine(text, cls) { $("engineText").textContent = text; $("engineDot").className = "dot " + cls; }
  const APP_VERSION = "2";

  // ---------------------------------------------------------------- state
  let info = null, result = null, charts = {}, running = false;
  const DEFAULT_MODELS = ["logreg", "linear", "rf", "lgbm"];
  const KIND_TAG = { id_like: "IDっぽい", constant: "値が1種類", empty: "空" };

  // ---------------------------------------------------------------- 1. data
  const drop = $("drop");
  ["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) readFile(f); });
  $("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) readFile(f); e.target.value = ""; });
  $("sampleBtn").addEventListener("click", async () => {
    try {
      const res = await fetch("sample/sample_customers.csv");
      if (!res.ok) throw new Error(res.status);
      loadText(await res.text(), "sample_customers.csv（合成データ）", "churned");
    } catch (e) { alertNote("サンプルを読み込めませんでした（" + e.message + "）"); }
  });

  // Excel で保存した日本語CSV（Shift_JIS）にも対応：UTF-8 で読めなければ Shift_JIS で読む
  async function readFile(f) {
    if (f.size > 50 * 1024 * 1024) { alertNote("50MBを超えるファイルはブラウザでは重すぎます。行を減らしてから試してください。"); return; }
    let text;
    try {
      const buf = await f.arrayBuffer();
      try { text = new TextDecoder("utf-8", { fatal: true }).decode(buf); }
      catch (_) { text = new TextDecoder("shift_jis").decode(buf); }
    } catch (e) { alertNote("ファイルを読めませんでした：" + e.message); return; }
    loadText(text, f.name, null);
  }

  async function loadText(text, name, preferTarget) {
    $("dataMeta").textContent = catalogue ? "読み込み中…" : "計算エンジンの準備ができたら読み込みます…";
    $("dataInfo").hidden = false;
    try {
      info = await call("load", { text });
      lastText = text; lastName = name;
    } catch (e) {
      $("dataMeta").textContent = "読み込めませんでした：" + e.message; return;
    }
    $("dataMeta").textContent = `${name}　${info.rows.toLocaleString()} 行 × ${info.cols.length} 列`;
    renderPreview();
    setupTarget(preferTarget);
    $("stepSetup").hidden = false; $("stepResult").hidden = true;
  }

  function renderPreview() {
    const t = $("previewTable"); t.replaceChildren();
    t.append(el("thead", {}, el("tr", {}, ...info.header.map((h) => el("th", { text: h })))));
    t.append(el("tbody", {}, ...info.preview.map((r) => el("tr", {}, ...r.map((v) => el("td", { text: v }))))));
  }

  // ---------------------------------------------------------------- 2. setup
  function setupTarget(prefer) {
    const sel = $("target"); sel.replaceChildren();
    for (const c of info.cols) sel.append(el("option", { value: c.name, text: c.name }));
    sel.value = prefer && info.cols.some((c) => c.name === prefer) ? prefer : info.cols[info.cols.length - 1].name;
    onTargetChange();
  }
  $("target").addEventListener("change", onTargetChange);
  $("task").addEventListener("change", () => { refreshModels(); updateTaskHint(); });

  function currentTask() {
    const v = $("task").value;
    if (v !== "auto") return v;
    const c = info && info.cols.find((x) => x.name === $("target").value);
    return c ? c.task : "classification";
  }
  function updateTaskHint() {
    const t = currentTask();
    $("taskHint").textContent = $("task").value === "auto"
      ? `→ ${t === "classification" ? "分類" : "回帰"}として扱います`
      : "";
  }
  function onTargetChange() {
    const tgt = $("target").value, box = $("features"); box.replaceChildren();
    for (const c of info.cols) {
      if (c.name === tgt) continue;
      const off = c.kind in KIND_TAG;
      box.append(el("label", { class: "chk" },
        el("input", { type: "checkbox", value: c.name, checked: !off, onchange: updateRunBtn }),
        c.name, off ? el("span", { class: "tag", text: KIND_TAG[c.kind] }) : null));
    }
    updateTaskHint(); refreshModels(); updateRunBtn();
  }
  $("featAll").addEventListener("click", () => { box("features").forEach((i) => (i.checked = true)); updateRunBtn(); });
  $("featNone").addEventListener("click", () => { box("features").forEach((i) => (i.checked = false)); updateRunBtn(); });
  const box = (id) => [...$(id).querySelectorAll("input[type=checkbox]")];

  function refreshModels() {
    if (!catalogue) return;
    const list = catalogue[info ? currentTask() : "classification"];
    let keep = new Set(box("models").filter((i) => i.checked).map((i) => i.value));
    if (![...keep].some((k) => list.some((x) => x.id === k))) keep = new Set();
    const m = $("models"); m.replaceChildren();
    for (const x of list) {
      if (x.id === "baseline") continue;
      const on = x.available && (keep.size ? keep.has(x.id) : DEFAULT_MODELS.includes(x.id));
      m.append(el("label", { class: "chk" + (x.available ? "" : " off"), title: x.available ? "" : "ブラウザ版では使えません（手元の Python で動かすと使えます）" },
        el("input", { type: "checkbox", value: x.id, checked: on, disabled: !x.available, onchange: updateRunBtn }), x.name,
        x.available ? null : el("span", { class: "tag", text: "ブラウザ版では不可" })));
    }
    updateRunBtn();
  }
  $("testSize").addEventListener("input", (e) => ($("testSizeOut").textContent = e.target.value + "%"));

  function updateRunBtn() {
    const ok = !running && catalogue && info && box("features").some((i) => i.checked) && box("models").some((i) => i.checked && !i.disabled);
    $("runBtn").disabled = !ok;
    if (!catalogue && !running) setRunStatus("計算エンジンの準備を待っています…");
  }
  function setRunStatus(text, spin) {
    $("runStatus").replaceChildren(...(spin ? [el("span", { class: "spin", "aria-hidden": "true" })] : []), text);
  }
  function showError(e) {
    const box = $("runError"); box.replaceChildren(el("strong", { text: "エラー：" }), e.message);
    if (e.detail) box.append(el("details", {}, el("summary", { text: "詳しい内容（スクショを送るときはここも開いて）" }), el("pre", { text: e.detail })));
    box.hidden = false;
  }

  $("runBtn").addEventListener("click", async () => {
    const args = {
      target: $("target").value,
      features: box("features").filter((i) => i.checked).map((i) => i.value),
      task: $("task").value,
      models: box("models").filter((i) => i.checked).map((i) => i.value),
      test_size: Number($("testSize").value) / 100,
      seed: Math.max(0, parseInt($("seed").value, 10) || 0),
      cv_folds: Number($("cv").value),
    };
    if (running) return;
    running = true; updateRunBtn(); $("runError").hidden = true;
    setRunStatus(`学習中…（${args.models.length + 1} モデル${args.cv_folds ? `、${args.cv_folds}分割の交差検証つき` : ""}）`, true);
    const t0 = performance.now();
    let msg = "";
    try {
      result = await call("run", args);
      msg = `完了（${((performance.now() - t0) / 1000).toFixed(1)} 秒）`;
      try { renderResult(); } catch (e) { e.detail = e.stack; showError(new Error("結果の表示に失敗しました：" + e.message)); }
    } catch (e) {
      msg = "うまくいきませんでした";
      showError(e);
    } finally { running = false; updateRunBtn(); setRunStatus(msg); }
  });

  // ---------------------------------------------------------------- 3. result
  const COLS = {
    classification: [["accuracy", "正解率"], ["balanced_accuracy", "バランス正解率"], ["f1_macro", "F1（マクロ）"], ["roc_auc", "AUC"]],
    regression: [["r2", "R²"], ["mae", "MAE（平均誤差）"], ["rmse", "RMSE"]],
  };
  const BEST_KEY = { classification: "balanced_accuracy", regression: "r2" };

  function renderResult() {
    const r = result, cls = r.task === "classification";
    $("stepResult").hidden = false;
    $("resMeta").textContent = `${cls ? "分類" : "回帰"}：「${r.target}」を ${r.features.length} 列から予測　学習 ${r.n_train} 行 ／ テスト ${r.n_test} 行`;
    const notes = $("resNotes"); notes.replaceChildren(...r.notes.map((n) => el("li", { text: n })));
    $("metricHint").textContent = cls
      ? "正解率はクラスの偏りに引っぱられます。偏りがあるときは「バランス正解率」と AUC を見てください。いちばん良い行（バランス正解率）を色付けしています。"
      : "R² は 1 に近いほど良く、0 なら平均値を当てるのと同じ。MAE・RMSE は小さいほど良い。いちばん良い行（R²）を色付けしています。";

    // metrics table
    const cols = COLS[r.task], bk = BEST_KEY[r.task];
    const best = r.results.filter((x) => x.id !== "baseline").reduce((a, b) => ((b[bk] ?? -1e9) > (a?.[bk] ?? -1e9) ? b : a), null);
    const t = $("metricsTable"); t.replaceChildren();
    const head = ["モデル", ...cols.map((c) => c[1])];
    if (r.cv_folds) head.push(`交差検証（${cls ? "正解率" : "R²"}・${r.cv_folds}分割）`);
    head.push("時間（秒）");
    t.append(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))));
    const tb = el("tbody");
    for (const x of r.results) {
      const tr = el("tr", { class: x === best ? "best" : x.id === "baseline" ? "base" : "" }, el("td", { text: x.name }));
      for (const [k] of cols) tr.append(el("td", { class: "n", text: fmt(x[k]) }));
      if (r.cv_folds) tr.append(el("td", { class: "n", text: x.cv_mean == null ? "—" : `${fmt(x.cv_mean)} ± ${fmt(x.cv_sd)}` }));
      tr.append(el("td", { class: "n", text: fmt(x.seconds) }));
      tb.append(tr);
    }
    t.append(tb);

    // score chart
    const series = cls ? [["accuracy", "正解率"], ["balanced_accuracy", "バランス正解率"]] : [["r2", "R²"]];
    drawBar("scoreChart", r.results.map((x) => x.name), series.map(([k, lab], i) => ({
      label: lab, data: r.results.map((x) => x[k]), backgroundColor: palette()[i],
    })), { min: cls ? 0 : undefined, max: cls ? 1 : undefined });

    // importance + confusion selectors
    const trained = r.results.filter((x) => x.id !== "baseline");
    fillSelect("impModel", trained, best && best.id);
    renderImportance();
    $("confBox").hidden = !cls;
    if (cls) { fillSelect("confModel", r.results, best && best.id); renderConfusion(); }

    renderPreds();
    $("stepResult").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function fillSelect(id, list, sel) {
    const s = $(id); s.replaceChildren(...list.map((x) => el("option", { value: x.id, text: x.name })));
    if (sel) s.value = sel;
  }
  $("impModel").addEventListener("change", renderImportance);
  $("confModel").addEventListener("change", renderConfusion);

  function renderImportance() {
    const imp = (result.importances[$("impModel").value] || []).slice(0, 15);
    drawBar("impChart", imp.map((d) => d.feature), [{ label: "成績の落ち幅", data: imp.map((d) => d.mean), backgroundColor: palette()[0] }], { horizontal: true });
  }

  function renderConfusion() {
    const x = result.results.find((m) => m.id === $("confModel").value), cl = result.classes, t = $("confTable");
    t.replaceChildren();
    if (!x || !x.confusion) return;
    t.append(el("thead", {}, el("tr", {}, el("th", { text: "正解 ＼ 予測" }), ...cl.map((c) => el("th", { text: c })))));
    t.append(el("tbody", {}, ...x.confusion.map((row, i) => el("tr", {}, el("th", { text: cl[i] }),
      ...row.map((v, j) => el("td", { class: "n" + (i === j ? " hit" : ""), text: v }))))));
  }

  function predColumns() {
    const r = result, cols = [["row", "CSVの行"], ["actual", "正解"]];
    for (const x of r.results) {
      cols.push([x.id, x.name]);
      if (r.predictions.length && (x.id + "__prob") in r.predictions[0])
        cols.push([x.id + "__prob", `${x.name}（「${r.classes[1]}」の確率）`]);
    }
    return cols;
  }
  function renderPreds() {
    const cols = predColumns(), rows = result.predictions.slice(0, 100), t = $("predTable");
    t.replaceChildren(el("thead", {}, el("tr", {}, ...cols.map((c) => el("th", { text: c[1] })))));
    t.append(el("tbody", {}, ...rows.map((p) => el("tr", {}, ...cols.map(([k]) => el("td", { class: typeof p[k] === "number" ? "n" : "", text: p[k] ?? "" }))))));
    $("predHint").textContent = result.predictions.length > 100 ? `画面には最初の100行だけ表示（全 ${result.predictions.length} 行はCSVに入ります）` : "";
  }
  $("dlBtn").addEventListener("click", () => {
    const cols = predColumns();
    const q = (v) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const csv = [cols.map((c) => q(c[1])).join(","), ...result.predictions.map((p) => cols.map(([k]) => q(p[k])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" }));
    const a = el("a", { href: url, download: `predictions_${result.target}.csv` }); document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  // ---------------------------------------------------------------- charts
  const palette = () => [getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#1f6f5c", "#d9822b"];
  function drawBar(id, labels, datasets, opt = {}) {
    if (typeof Chart === "undefined") return;
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--muted").trim(), grid = css.getPropertyValue("--line").trim();
    charts[id]?.destroy();
    const valueAxis = { min: opt.min, max: opt.max, ticks: { color: ink }, grid: { color: grid } };
    const catAxis = { ticks: { color: ink, autoSkip: false }, grid: { display: false } };
    charts[id] = new Chart($(id), {
      type: "bar",
      data: { labels, datasets: datasets.map((d) => ({ ...d, borderRadius: 4, maxBarThickness: 36 })) },
      options: {
        indexAxis: opt.horizontal ? "y" : "x", responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: datasets.length > 1, labels: { color: ink } } },
        scales: opt.horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
      },
    });
  }

  const fmt = (v) => (v == null ? "—" : typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toLocaleString() : v.toFixed(3)) : v);
  function alertNote(t) { $("dataInfo").hidden = false; $("dataMeta").textContent = t; }
  startWorker();
})();
