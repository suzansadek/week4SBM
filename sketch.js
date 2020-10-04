// https://kylemcdonald.github.io/cv-examples/

var capture;
var previousPixels;
var flow;
var w = 640;
var h = 480;
var step = 8;
var avgOpticalFlow;
var smoothedAvgOpticalFlow;
var movers = [];
var numMovers = 1;


function setup() {
  createCanvas(w, h);
  capture = createCapture({
    audio: false,
    video: {
      width: w,
      height: h
    }
  }, function() {
    console.log('capture ready.')
  });
  capture.elt.setAttribute('playsinline', '');
  // capture.hide()
  capture.size(w, h);
  //   createCanvas(w, h);
  // let cnv = createCanvas(w, h);
  // positions canvas 50 px to the right and 100 px
  // below upper left corner of the window
  // cnv.position(360, 20, 'fixed');

  capture.hide();
  flow = new FlowCalculator(step);
  avgOpticalFlow = createVector(0, 0);
  smoothedAvgOpticalFlow = createVector(0, 0);

  for (var i = 0; i < numMovers; i++) {
    c = color(255, 100);
    var s = 3;
    var x = random(width);
    var y = random(height);
    movers[i] = new Mover(s, x, y, c);
  }

}

var backgroundPixels;

function resetBackground() {
  backgroundPixels = undefined;
}

function copyImage(src, dst) {
  var n = src.length;
  if (!dst || dst.length != n) {
    dst = new src.constructor(n);
  }
  while (n--) {
    dst[n] = src[n];
  }
  return dst;
}

var captureMat, gray, blurred, thresholded;
var contours, hierarchy;

function cvSetup() {
  captureMat = new cv.Mat([h, w], cv.CV_8UC4);
  gray = new cv.Mat([h, w], cv.CV_8UC1);
  blurred = new cv.Mat([h, w], cv.CV_8UC1);
  thresholded = new cv.Mat([h, w], cv.CV_8UC1);
}

var ready = false;

function cvReady() {
  if (!cv || !cv.loaded) return false;
  if (ready) return true;
  cvSetup();
  ready = true;
  return true;
}

function same(a1, a2, stride, n) {
  for (var i = 0; i < n; i += stride) {
    if (a1[i] != a2[i]) {
      return false;
    }
  }
}

function draw() {

  var blurRadius = select('#blurRadius').value();
  blurRadius = map(blurRadius, 0, 100, 1, 20);

  var threshold = select('#threshold').value();
  threshold = map(threshold, 0, 100, 0, 255);

  var minArea = select('#minArea').value();
  minArea = map(minArea, 0, 100, 0, 200000);

  var maxArea = select('#maxArea').value();
  maxArea = map(maxArea, 0, 100, 0, 200000);

  var showThresholded = select('#showThresholded').checked();

  if (cvReady()) {
    capture.loadPixels();
    if (capture.pixels.length > 0) {
      // if (!backgroundPixels) {
      //   backgroundPixels = copyImage(capture.pixels, backgroundPixels);
      //
      //   if (same(previousPixels, capture.pixels, 4, width)) {
      //     return;
      //   }
      //   flow.calculate(previousPixels, capture.pixels, capture.width, capture.height);
      // }

      if (!backgroundPixels) {
        backgroundPixels = copyImage(capture.pixels, backgroundPixels);
      }

      if (previousPixels) {
        // cheap way to ignore duplicate frames
        if (same(previousPixels, capture.pixels, 4, width)) {
          return;
        }
        flow.calculate(previousPixels, capture.pixels, capture.width, capture.height);
      }
      previousPixels = copyImage(capture.pixels, previousPixels);




      var i = 0;
      var pixels = capture.pixels;
      var total = 0;
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          var rdiff = Math.abs(pixels[i + 0] - backgroundPixels[i + 0]) > threshold;
          var gdiff = Math.abs(pixels[i + 1] - backgroundPixels[i + 1]) > threshold;
          var bdiff = Math.abs(pixels[i + 2] - backgroundPixels[i + 2]) > threshold;
          var anydiff = rdiff || gdiff || bdiff;
          var output = 0;
          if (anydiff) {
            output = 255;
            total++;
          }
          pixels[i++] = output;
          pixels[i++] = output;
          pixels[i++] = output;
          i++; // skip alpha
        }
      }
      captureMat.data().set(pixels);

      cv.cvtColor(captureMat, gray, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
      cv.blur(gray, blurred, [blurRadius, blurRadius], [-1, -1], cv.BORDER_DEFAULT);
      cv.threshold(blurred, thresholded, threshold, 255, cv.ThresholdTypes.THRESH_BINARY.value);

      if (showThresholded) {
        var src = thresholded.data();
        var dst = capture.pixels;
        var n = src.length;
        var j = 0;
        for (var i = 0; i < n; i++) {
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = 255;
        }
        capture.updatePixels();
      }
      image(capture, 0, 0, w, h);


      contours = new cv.MatVector();
      hierarchy = new cv.Mat();
      cv.findContours(thresholded, contours, hierarchy, 3, 2, [0, 0]);

      previousPixels = copyImage(capture.pixels, previousPixels);
      // image(capture, 0, 0, w, h);

      if (flow.flow && flow.flow.u != 0 && flow.flow.v != 0) {
        strokeWeight(2);
        // u and V are used as labers for vector coordinates

        flow.flow.zones.forEach(function(zone) {
          stroke(map(zone.u, -step, +step, 0, 255),
            map(zone.v, -step, +step, 0, 255), 128);
          // drawing all the lines, X and Y location + Direction
          line(zone.x, zone.y, zone.x + zone.u, zone.y + zone.v);
        })
        // do smoothing \\

        // create a vecot first with the average optical flow
        avgOpticalFlow = createVector(flow.flow.u, flow.flow.v);
        // copy our current smoothed value into a new vector called smoothing
        var smoothing = avgOpticalFlow.copy();

        // remember, our smoothing algorithm is take where we are currently and...
        // ADD our destination minus where we are currently times a scalar
        // i.e. smoothedValue += (destination - smoothedValue) * .1

        // this is the first part: subtract where we are from our destination
        // (remember, we copied the new average flow into the smoothing variable)
        smoothing.sub(smoothedAvgOpticalFlow);
        // then multiply that by a scalar
        smoothing.mult(.1);
        // then add that to where we are currently
        smoothedAvgOpticalFlow.add(smoothing);

        for (var i = 0; i < numMovers; i++) {
          var arrayLoc = flow.getArrayLoc(movers[i].position.x, movers[i].position.y, w);
          var zone = flow.flow.zones[arrayLoc];
          // console.log(flow.flow.zones[arrayLoc]);
          var force;
          if (zone) {
            force = createVector(zone.u, zone.v);
            movers[i].applyForces(force);
          }
          movers[i].update();

          movers[i].display();
        }
        console.log("x: " + avgOpticalFlow.x + "\ny: " + avgOpticalFlow.y + "\nlength: " + smoothedAvgOpticalFlow.mag());

        sendOsc('/ctrl', 'optflowX', smoothedAvgOpticalFlow.x);
        sendOsc('/ctrl', 'optflowY', smoothedAvgOpticalFlow.y);
        sendOsc('/ctrl', 'optflowLength', smoothedAvgOpticalFlow.mag());

        var moverX = map(movers[0].position.x, 0, w, 0., 1.);
        sendOsc('/ctrl', 'moverX', moverX);
        var moverY = map(movers[0].position.y, 0, h, 0., 1.);
        sendOsc('/ctrl', 'moverY', moverY);

        // draw the average flow
        strokeWeight(5);
        stroke(255, 0, 0);
        // multipling by 10 in order to see on screen better
        line(width / 2, height / 2, width / 2 + smoothedAvgOpticalFlow.x * 10, height / 2 + smoothedAvgOpticalFlow.y * 10);
      }

      // image(capture, 0, 0, w, h);

      if (contours && !showThresholded) {
        noStroke();
        for (var i = 0; i < contours.size(); i++) {
          fill(0, 0, 255, 128);
          var contour = contours.get(i);
          var area = cv.contourArea(contour, false);

          if (area > minArea && area < maxArea) {
            beginShape();
            var k = 0;
            for (var j = 0; j < contour.total(); j++) {
              var x = contour.get_int_at(k++);
              var y = contour.get_int_at(k++);
              vertex(x, y);
            }
            endShape(CLOSE);

            noFill();
            stroke(255, 255, 255)
            var box = cv.boundingRect(contour);
            var center = createVector(box.x + box.width / 2, box.y + box.height / 2);
            ellipse(center.x, center.y, 10, 10);
            rect(box.x, box.y, box.width, box.height);

            // send OSC
            var x = map(center.x, 0, width, 0, 1);
            sendOsc('/ctrl', 'speed', x);
            var y = map(center.y, 0, height, 0, 1);
            sendOsc('/ctrl', 'gain', y);

          }

        }
      }
    }
  }
}
