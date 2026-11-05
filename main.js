import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

let poseLandmarker;
let video = document.getElementById("webcam");
let canvas = document.getElementById("output_canvas");
let ctx = canvas.getContext("2d");

const params = new URLSearchParams(window.location.search);
const numPoses = params.get("numPoses") ? parseInt(params.get("numPoses")) : 1;
const lineWidth = params.get("lineWidth")
  ? parseInt(params.get("lineWidth"))
  : 8;

// 1. MediaPipeの初期化
async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    numPoses,
  });

  startCamera();
}

// 2. カメラの起動
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.addEventListener("loadeddata", predictWebcam);
}

// 3. リアルタイム検出ループ
let lastVideoTime = -1;
async function predictWebcam() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = poseLandmarker.detectForVideo(video, performance.now());

    ctx.fillStyle = "blue";
    ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fill();
    for (const landmark of results.landmarks) {
      drawStickman(ctx, landmark);
    }
  }
  requestAnimationFrame(predictWebcam);
}

function drawStickman(ctx, points) {
  if (!points || points.length === 0) return;
  const leftHip = points[23];
  const rightHip = points[24];
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
  };

  const headCenter = points[0];

  const leftShoulder = points[11];
  const rightShoulder = points[12];

  const shoulderWidth = Math.sqrt(
    Math.pow(leftShoulder.x - rightShoulder.x, 2) +
      Math.pow(leftShoulder.y - rightShoulder.y, 2),
  );

  const crossPoint = calcCrossPoint(
    leftShoulder,
    rightShoulder,
    headCenter,
    hipCenter,
  );
  const crossPoint2leftShoulder = subtractPoints(leftShoulder, crossPoint);
  const crossPoint2rightShoulder = subtractPoints(rightShoulder, crossPoint);

  const leftElbow = subtractPoints(points[13], crossPoint2leftShoulder);
  const rightElbow = subtractPoints(points[14], crossPoint2rightShoulder);
  const leftWrist = subtractPoints(points[15], crossPoint2leftShoulder);
  const rightWrist = subtractPoints(points[16], crossPoint2rightShoulder);

  ctx.fillStyle = "white";
  ctx.strokeStyle = "black";
  ctx.lineWidth = lineWidth;

  const drawLeftArm = () => {
    ctx.beginPath();
    ctx.moveTo(crossPoint.x * canvas.width, crossPoint.y * canvas.height);
    ctx.lineTo(leftElbow.x * canvas.width, leftElbow.y * canvas.height);
    ctx.lineTo(leftWrist.x * canvas.width, leftWrist.y * canvas.height);
    ctx.stroke();
  };
  const drawRightArm = () => {
    ctx.beginPath();
    ctx.moveTo(crossPoint.x * canvas.width, crossPoint.y * canvas.height);
    ctx.lineTo(rightElbow.x * canvas.width, rightElbow.y * canvas.height);
    ctx.lineTo(rightWrist.x * canvas.width, rightWrist.y * canvas.height);
    ctx.stroke();
  };
  const drawBody = () => {
    ctx.beginPath();
    ctx.moveTo(headCenter.x * canvas.width, headCenter.y * canvas.height);
    ctx.lineTo(hipCenter.x * canvas.width, hipCenter.y * canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(
      headCenter.x * canvas.width,
      headCenter.y * canvas.height,
      shoulderWidth * 0.5 * canvas.width,
      0,
      2 * Math.PI,
    );
    ctx.fill();
    ctx.stroke();
  };

  drawLeftArm();
  drawRightArm();
  drawBody();
}

// p1 -> p2 と p3 -> p4 の交点を求める
function calcCrossPoint(p1, p2, p3, p4) {
  const devider = (p2.y - p1.y) * (p3.x - p4.x) - (p1.x - p2.x) * (p4.y - p3.y);
  const xDevidend =
    (p3.x - p4.x) * (p1.x * p2.y - p2.x * p1.y) +
    (p2.x - p1.x) * (p3.x * p4.y - p4.x * p3.y);
  const yDevidend =
    (p3.y - p4.y) * (p1.x * p2.y - p2.x * p1.y) +
    (p2.y - p1.y) * (p3.x * p4.y - p4.x * p3.y);
  return { x: xDevidend / devider, y: yDevidend / devider };
}

function subtractPoints(p1, p2) {
  return { ...p1, ...p2, x: p1.x - p2.x, y: p1.y - p2.y };
}

initMediaPipe();

window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});
