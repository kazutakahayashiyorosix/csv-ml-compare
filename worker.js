/* 計算を担当する Web Worker。Pyodide（ブラウザで動く Python）で scikit-learn を動かす。
   画面とは別スレッドなので、学習中も画面が固まらない。 */
const PYODIDE_VERSION = "0.26.4";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

let core = null;

async function init() {
  postMessage({ type: "status", text: "Python を読み込み中…" });
  const py = await loadPyodide();
  postMessage({ type: "status", text: "numpy・pandas・scikit-learn を読み込み中…（初回は少し時間がかかります）" });
  await py.loadPackage(["numpy", "pandas", "scikit-learn"]);
  const src = await (await fetch("ml_core.py", { cache: "no-cache" })).text();
  py.FS.writeFile("ml_core.py", src);
  py.runPython("import sys; sys.path.insert(0, '.')");
  core = py.pyimport("ml_core");
  postMessage({ type: "ready", catalogue: JSON.parse(core.catalogue()) });
}

const ready = init().catch((e) => postMessage({ type: "fatal", text: String(e) }));

onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  await ready;
  try {
    let out;
    if (cmd === "load") out = core.load_csv(args.text);
    else if (cmd === "run")
      out = core.run(args.target, args.features, args.task, args.models,
                     args.test_size, args.seed, args.cv_folds);
    else throw new Error("unknown command " + cmd);
    postMessage({ id, ok: true, data: JSON.parse(out) });
  } catch (e) {
    // Python の例外は最後の行がメッセージ
    const msg = String(e && e.message ? e.message : e).trim().split("\n").pop();
    postMessage({ id, ok: false, error: msg.replace(/^\w+Error:\s*/, "") });
  }
};
