/* 計算を担当する Web Worker。Pyodide（ブラウザで動く Python）で scikit-learn などを動かす。
   画面とは別スレッドなので、学習中も画面が固まらない。 */
const PYODIDE_VERSION = "0.28.2";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const LOADABLE = ["lightgbm", "xgboost"];   // 必要になった時だけ読み込む（CatBoost はブラウザ版なし）

let py = null, core = null;
const loaded = new Set();

async function init() {
  postMessage({ type: "status", text: "Python を読み込み中…" });
  importScripts(INDEX_URL + "pyodide.js");
  py = await loadPyodide({ indexURL: INDEX_URL });
  postMessage({ type: "status", text: "numpy・pandas・scikit-learn を読み込み中…（初回は少し時間がかかります）" });
  await py.loadPackage(["numpy", "pandas", "scikit-learn"]);
  const res = await fetch("ml_core.py?v=2", { cache: "no-cache" });
  if (!res.ok) throw new Error("ml_core.py を読み込めません（" + res.status + "）");
  py.FS.writeFile("ml_core.py", await res.text());
  py.runPython("import sys, importlib; sys.path.insert(0, '.'); import ml_core; importlib.reload(ml_core)");
  core = py.pyimport("ml_core");
  postMessage({ type: "ready", catalogue: JSON.parse(core.catalogue(py.toPy(LOADABLE))) });
}

const ready = init();
ready.catch((e) => postMessage({ type: "fatal", text: pyMsg(e) }));

function pyMsg(e) {
  const full = String(e && e.message ? e.message : e).trim();
  const last = full.split("\n").filter((l) => l.trim()).pop() || full;
  return last.replace(/^\w+(Error|Exception):\s*/, "");
}

async function ensurePackages(task, models) {
  const need = JSON.parse(core.required_packages(task, py.toPy(models))).filter((p) => LOADABLE.includes(p) && !loaded.has(p));
  if (!need.length) return;
  postMessage({ type: "progress", text: need.join("・") + " を読み込み中…（初回のみ）" });
  await py.loadPackage(need);
  need.forEach((p) => loaded.add(p));
}

onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  try {
    await ready;
  } catch (e) {
    postMessage({ id, ok: false, error: "計算エンジンを起動できませんでした：" + pyMsg(e) });
    return;
  }
  try {
    let out;
    if (cmd === "load") out = core.load_csv(args.text);
    else if (cmd === "run") {
      await ensurePackages(args.task, args.models);
      out = core.run(args.target, py.toPy(args.features), args.task, py.toPy(args.models),
                     args.test_size, args.seed, args.cv_folds);
    } else throw new Error("unknown command " + cmd);
    postMessage({ id, ok: true, data: JSON.parse(out) });
  } catch (e) {
    postMessage({ id, ok: false, error: pyMsg(e), detail: String(e && e.message ? e.message : e).slice(-3000) });
  }
};
