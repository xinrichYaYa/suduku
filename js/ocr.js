// Simple client-side OCR loader for Sudoku images
// Requires Tesseract.js to be loaded first (we include via CDN in index.html)
/** @type {any} */
var Tesseract;
/** @type {any} */
var cv;

(function() {
  // helper: draw image to canvas and scale to target size while preserving aspect
  function drawImageToCanvas(img, canvas) {
    var ctx = canvas.getContext('2d');
    // clear
    ctx.clearRect(0,0,canvas.width, canvas.height);
    // fit image into canvas
    var cw = canvas.width, ch = canvas.height;
    var iw = img.width, ih = img.height;
    var scale = Math.min(cw/iw, ch/ih);
    var nw = iw * scale, nh = ih * scale;
    var dx = (cw - nw)/2, dy = (ch - nh)/2;
    ctx.fillStyle = '#fff'; ctx.fillRect(0,0,cw,ch);
    ctx.drawImage(img, 0,0,iw,ih, dx, dy, nw, nh);
    return {dx:dx, dy:dy, nw:nw, nh:nh};
  }

  // preprocess a cell canvas: convert to grayscale and threshold
  function preprocessCell(cellCanvas) {
    var w = cellCanvas.width, h = cellCanvas.height;
    var ctx = cellCanvas.getContext('2d');
    var imgd = ctx.getImageData(0,0,w,h);
    var data = imgd.data;
    // simple grayscale + binary threshold
    for (var i=0;i<data.length;i+=4) {
      var r = data[i], g = data[i+1], b = data[i+2];
      var gray = (r*0.3 + g*0.59 + b*0.11)|0;
      var th = gray > 200 ? 255 : 0; // adjust threshold for light backgrounds
      data[i]=data[i+1]=data[i+2]=th;
      // keep alpha
    }
    ctx.putImageData(imgd, 0,0);
  }

  async function recognizeGridFromCanvas(canvas, statusEl) {
    var ctx = canvas.getContext('2d');
    var BoardSize = 9;
    var cellW = Math.floor(canvas.width / BoardSize);
    var cellH = Math.floor(canvas.height / BoardSize);
    var resultChars = [];

    // Helper: quickly estimate whether a cell is essentially blank (few dark pixels)
    function isCellBlank(cvs) {
      try {
        var w = cvs.width, h = cvs.height;
        var ct = cvs.getContext('2d');
        var imgd = ct.getImageData(0,0,w,h);
        var data = imgd.data;
        var dark = 0, total = w*h;
        for (var i=0;i<data.length;i+=4) {
          // consider pixel dark if grayscale < 220
          var gray = (data[i]*0.3 + data[i+1]*0.59 + data[i+2]*0.11)|0;
          if (gray < 220) dark++;
        }
        // if less than ~2% of pixels are dark, treat as blank
        return (dark / total) < 0.02;
      } catch (e) {
        return false;
      }
    }

    // Use a single Tesseract worker to avoid repeated startup cost
    var worker = null;
    try {
      worker = Tesseract.createWorker({
        logger: function(m) { /* optional logging: console.log(m) */ }
      });
      await worker.load();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');
      // single-character mode + whitelist improves speed/accuracy for digits
      await worker.setParameters({ tessedit_char_whitelist: '0123456789', tessedit_pageseg_mode: '10' });
    } catch (e) {
      console.warn('Tesseract worker init failed, falling back to recognize()', e);
      if (worker) try { await worker.terminate(); } catch(e){}
      worker = null;
    }

    // sequential recognition per cell, but with fast blank-skip and downscaling
    for (var r=0;r<BoardSize;r++){
      for (var c=0;c<BoardSize;c++){
        // crop a little inset to avoid grid lines
        var insetW = Math.floor(cellW*0.12);
        var insetH = Math.floor(cellH*0.12);
        var sx = c*cellW + insetW;
        var sy = r*cellH + insetH;
        var sw = Math.max(8, cellW - insetW*2);
        var sh = Math.max(8, cellH - insetH*2);
        // temp canvas
        var cellCanvas = document.createElement('canvas');
        cellCanvas.width = sw; cellCanvas.height = sh;
        var cellCtx = cellCanvas.getContext('2d');
        cellCtx.drawImage(canvas, sx, sy, sw, sh, 0,0, sw, sh);
        // quick blank check before full preprocessing / OCR
        if (isCellBlank(cellCanvas)) {
          resultChars.push('.');
          statusEl.innerText = `跳过空白 ${r+1},${c+1}`;
          continue;
        }

        // Preprocess: binary thresholding
        preprocessCell(cellCanvas);

        // Downscale to small fixed size to speed recognition (keeps digit shape)
        var small = document.createElement('canvas');
        small.width = 64; small.height = 64;
        var sctx = small.getContext('2d');
        sctx.fillStyle = '#fff'; sctx.fillRect(0,0,small.width, small.height);
        sctx.drawImage(cellCanvas, 0,0, cellCanvas.width, cellCanvas.height, 0,0, small.width, small.height);

        statusEl.innerText = `识别行 ${r+1} 列 ${c+1} ...`;
        try {
          var res = null;
          if (worker) {
            res = await worker.recognize(small);
          } else {
            // fallback to direct API
            res = await Tesseract.recognize(small, 'eng', { tessedit_char_whitelist: '0123456789' });
          }
          var text = (res && res.data && res.data.text) ? res.data.text.replace(/\s+/g,'') : '';
          var ch = text.length>0 ? text[0] : '.';
          if (!/^[0-9]$/.test(ch)) ch='.';
          resultChars.push(ch);
        } catch(e) {
          console.error('OCR cell error', e);
          resultChars.push('.');
        }
      }
    }

    if (worker) try { await worker.terminate(); } catch (e) {}
    statusEl.innerText = '识别完成';
    return resultChars.join('');
  }

  // If OpenCV.js is available, try to detect the largest square contour (the sudoku grid)
  // and warp it to a square canvas. Returns a Promise that resolves to a canvas element
  // containing the warped/normalized grid (same size as destCanvas if provided) or null on failure.
  function detectAndWarpGrid(sourceCanvas, destWidth, destHeight, statusEl) {
    return new Promise(function(resolve) {
      if (typeof cv === 'undefined' || !cv || !cv.Mat) {
        // OpenCV not available
        resolve(null);
        return;
      }
      try {
        // Read source into cv.Mat
        var src = cv.imread(sourceCanvas);
        var gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        var blurred = new cv.Mat();
        var ksize = new cv.Size(5,5);
        cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT);
        var thresh = new cv.Mat();
        cv.adaptiveThreshold(blurred, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

        // find contours
        var contours = new cv.MatVector();
        var hierarchy = new cv.Mat();
        cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        var maxArea = 0;
        var bestContour = null;
        for (var i = 0; i < contours.size(); ++i) {
          var cnt = contours.get(i);
          var area = cv.contourArea(cnt, false);
          if (area > maxArea) {
            // approximate polygon
            var peri = cv.arcLength(cnt, true);
            var approx = new cv.Mat();
            cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
            if (approx.rows === 4 && area > maxArea) {
              maxArea = area;
              bestContour = approx; // note: approx is a Mat, keep reference
            } else {
              approx.delete();
            }
          }
          cnt.delete();
        }

        if (!bestContour) {
          // cleanup
          src.delete(); gray.delete(); blurred.delete(); thresh.delete(); contours.delete(); hierarchy.delete();
          resolve(null);
          return;
        }

        // extract 4 points from bestContour (use data32S which stores [x0,y0,x1,y1,...])
        var pts = [];
        var data = bestContour.data32S;
        for (var i = 0; i < 4; i++) {
          pts.push({x: data[i*2], y: data[i*2 + 1]});
        }

        // order points: top-left, top-right, bottom-right, bottom-left
        pts.sort(function(a,b){ return a.y - b.y; }); // sort by y
        var top = pts.slice(0,2).sort(function(a,b){ return a.x - b.x; });
        var bottom = pts.slice(2,4).sort(function(a,b){ return a.x - b.x; });
        var ordered = [ top[0], top[1], bottom[1], bottom[0] ];

        // prepare source and destination points
        var srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
          ordered[0].x, ordered[0].y,
          ordered[1].x, ordered[1].y,
          ordered[2].x, ordered[2].y,
          ordered[3].x, ordered[3].y
        ]);
        var dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
          0, 0,
          destWidth-1, 0,
          destWidth-1, destHeight-1,
          0, destHeight-1
        ]);

        var M = cv.getPerspectiveTransform(srcTri, dstTri);
        var dst = new cv.Mat();
        var dsize = new cv.Size(destWidth, destHeight);
        cv.warpPerspective(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

        // copy dst to output canvas
        var outCanvas = document.createElement('canvas');
        outCanvas.width = destWidth; outCanvas.height = destHeight;
        cv.imshow(outCanvas, dst);

        // cleanup
        src.delete(); gray.delete(); blurred.delete(); thresh.delete(); contours.delete(); hierarchy.delete();
        bestContour.delete(); srcTri.delete(); dstTri.delete(); M.delete(); dst.delete();

        resolve(outCanvas);
      } catch (e) {
        console.error('OpenCV grid detection error', e);
        resolve(null);
      }
    });
  }
})();
