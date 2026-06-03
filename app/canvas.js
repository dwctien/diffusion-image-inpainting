// ── Mask Drawing Canvas ──────────────────────────────────────────
// Three-layer system:
//   bgCanvas   – original image at full resolution
//   maskCanvas – mask on TRANSPARENT background (white = mask, clear = keep)
//   canvas     – visible display compositing bg (dimmed) + mask (red)

window.MaskPainter = (function () {
  let canvas, ctx;
  let bgCanvas, bgCtx;
  let maskCanvas, maskCtx;
  let tool = "brush";
  let brushSize = 30;
  let drawing = false;
  let startX = 0, startY = 0;
  let lastX = 0, lastY = 0;
  let imageLoaded = false;
  let snapshotData = null;
  let initDone = false;

  /* ── coordinate helpers ─────────────────────────────────── */
  function pos(e) {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width;
    const sy = canvas.height / r.height;
    const src = e.touches ? e.touches[0] : e;
    return { x: (src.clientX - r.left) * sx, y: (src.clientY - r.top) * sy };
  }

  /* ── redraw composite ──────────────────────────────────── */
  function redraw() {
    if (!canvas) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // 1) background image at reduced opacity
    if (imageLoaded) {
      ctx.globalAlpha = 0.45;
      ctx.drawImage(bgCanvas, 0, 0);
      ctx.globalAlpha = 1.0;
    }

    // 2) color the mask: white-on-transparent → red overlay
    //    Using an offscreen canvas with source-in compositing
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    const tc = tmp.getContext("2d");
    tc.drawImage(maskCanvas, 0, 0);            // white on transparent
    tc.globalCompositeOperation = "source-in";  // keep only opaque parts
    tc.fillStyle = "rgba(231, 76, 60, 0.6)";
    tc.fillRect(0, 0, w, h);                   // opaque white → red

    ctx.drawImage(tmp, 0, 0);
  }

  /* ── draw helpers ──────────────────────────────────────── */
  function brushStroke(x1, y1, x2, y2) {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.strokeStyle = "#fff";
    maskCtx.lineWidth = brushSize;
    maskCtx.lineCap = "round";
    maskCtx.lineJoin = "round";
    maskCtx.beginPath();
    maskCtx.moveTo(x1, y1);
    maskCtx.lineTo(x2, y2);
    maskCtx.stroke();
  }

  function brushDot(x, y) {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    maskCtx.fillStyle = "#fff";
    maskCtx.fill();
  }

  function eraseDot(x, y) {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    maskCtx.fillStyle = "rgba(0,0,0,1)";
    maskCtx.fill();
    maskCtx.globalCompositeOperation = "source-over";
  }

  function eraseStroke(x1, y1, x2, y2) {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.strokeStyle = "rgba(0,0,0,1)";
    maskCtx.lineWidth = brushSize;
    maskCtx.lineCap = "round";
    maskCtx.lineJoin = "round";
    maskCtx.beginPath();
    maskCtx.moveTo(x1, y1);
    maskCtx.lineTo(x2, y2);
    maskCtx.stroke();
    maskCtx.globalCompositeOperation = "source-over";
  }

  /* ── init ──────────────────────────────────────────────── */
  function init() {
    canvas = document.getElementById("mask-draw-canvas");
    if (!canvas) { setTimeout(init, 300); return; }
    if (initDone) return;
    initDone = true;

    ctx = canvas.getContext("2d");
    bgCanvas  = document.createElement("canvas");
    bgCtx     = bgCanvas.getContext("2d");
    maskCanvas = document.createElement("canvas");
    maskCtx    = maskCanvas.getContext("2d");

    // default placeholder size
    canvas.width = 512; canvas.height = 340;
    bgCanvas.width = 512; bgCanvas.height = 340;
    maskCanvas.width = 512; maskCanvas.height = 340;
    showPlaceholder();

    // toolbar buttons
    document.querySelectorAll(".mp-tool").forEach(function(btn) {
      btn.addEventListener("click", function() {
        document.querySelectorAll(".mp-tool").forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        tool = btn.dataset.tool;
        canvas.style.cursor = tool === "eraser" ? "cell" : "crosshair";
      });
    });
    var slider = document.getElementById("mp-brush-size");
    var sizeLabel = document.getElementById("mp-size-label");
    if (slider) slider.addEventListener("input", function() {
      brushSize = parseInt(slider.value);
      if (sizeLabel) sizeLabel.textContent = brushSize + "px";
    });
    var clearBtn = document.getElementById("mp-clear");
    if (clearBtn) clearBtn.addEventListener("click", clearMask);

    // pointer events
    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", onUp);
    canvas.addEventListener("touchstart", function(e) { e.preventDefault(); onDown(e); }, { passive: false });
    canvas.addEventListener("touchmove",  function(e) { e.preventDefault(); onMove(e); }, { passive: false });
    canvas.addEventListener("touchend",   function(e) { e.preventDefault(); onUp(e);   }, { passive: false });
  }

  function showPlaceholder() {
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#64748b";
    ctx.font = "15px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Upload ảnh ở Bước 1 để bắt đầu vẽ mask", canvas.width / 2, canvas.height / 2);
  }

  /* ── load background image ─────────────────────────────── */
  function loadImage(b64) {
    if (!b64 || !canvas) return;
    var img = new Image();
    img.onload = function() {
      var w = img.width, h = img.height;
      canvas.width = w;     canvas.height = h;
      bgCanvas.width = w;   bgCanvas.height = h;
      maskCanvas.width = w;  maskCanvas.height = h;
      // draw background
      bgCtx.drawImage(img, 0, 0);
      // mask starts fully transparent (no mask)
      maskCtx.clearRect(0, 0, w, h);
      imageLoaded = true;
      // responsive display
      var maxW = 580;
      var scale = Math.min(maxW / w, 1);
      canvas.style.width  = Math.round(w * scale) + "px";
      canvas.style.height = Math.round(h * scale) + "px";
      redraw();
    };
    img.src = b64;
  }

  /* ── clear ─────────────────────────────────────────────── */
  function clearMask() {
    if (!imageLoaded) return;
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    redraw();
  }

  /* ── export: white-on-black PNG ────────────────────────── */
  function exportMask() {
    if (!imageLoaded) return "";
    var exp = document.createElement("canvas");
    exp.width = maskCanvas.width; exp.height = maskCanvas.height;
    var ec = exp.getContext("2d");
    ec.fillStyle = "#000";
    ec.fillRect(0, 0, exp.width, exp.height);
    ec.drawImage(maskCanvas, 0, 0);
    return exp.toDataURL("image/png");
  }

  /* ── event handlers ────────────────────────────────────── */
  function onDown(e) {
    if (!imageLoaded) return;
    drawing = true;
    var p = pos(e);
    startX = p.x; startY = p.y;
    lastX  = p.x; lastY  = p.y;

    if (tool === "brush") {
      brushDot(p.x, p.y);
      redraw();
    } else if (tool === "eraser") {
      eraseDot(p.x, p.y);
      redraw();
    } else {
      // rect / circle: snapshot mask for live preview
      snapshotData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    }
  }

  function onMove(e) {
    if (!drawing || !imageLoaded) return;
    var p = pos(e);

    if (tool === "brush") {
      brushStroke(lastX, lastY, p.x, p.y);
      lastX = p.x; lastY = p.y;
      redraw();
    } else if (tool === "eraser") {
      eraseStroke(lastX, lastY, p.x, p.y);
      lastX = p.x; lastY = p.y;
      redraw();
    } else {
      // rect / circle: restore snapshot then draw shape
      if (snapshotData) maskCtx.putImageData(snapshotData, 0, 0);
      maskCtx.globalCompositeOperation = "source-over";
      maskCtx.fillStyle = "#fff";
      if (tool === "rect") {
        maskCtx.fillRect(
          Math.min(startX, p.x), Math.min(startY, p.y),
          Math.abs(p.x - startX), Math.abs(p.y - startY)
        );
      } else if (tool === "circle") {
        var rx = Math.abs(p.x - startX) / 2;
        var ry = Math.abs(p.y - startY) / 2;
        var cx = (startX + p.x) / 2;
        var cy = (startY + p.y) / 2;
        maskCtx.beginPath();
        maskCtx.ellipse(cx, cy, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2);
        maskCtx.fill();
      }
      redraw();
    }
  }

  function onUp() {
    drawing = false;
    snapshotData = null;
  }

  /* ── public API ────────────────────────────────────────── */
  return { init: init, loadImage: loadImage, exportMask: exportMask, clearMask: clearMask };
})();

// auto-init (with retry)
(function tryInit() {
  if (document.getElementById("mask-draw-canvas")) {
    window.MaskPainter.init();
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function() { window.MaskPainter.init(); });
  } else {
    setTimeout(tryInit, 400);
  }
})();
