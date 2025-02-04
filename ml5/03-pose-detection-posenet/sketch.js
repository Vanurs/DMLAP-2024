/*
  Data and Machine Learning for Artistic Practice (DMLAP)
  Spring Term 2024 
  Pose Estimation with PoseNet

  Originally from: The Coding Train / Daniel Shiffman
  https://thecodingtrain.com/Courses/ml5-beginners-guide/7.1-posenet.html
  https://youtu.be/OIo-DIOkNVg
  https://editor.p5js.org/codingtrain/sketches/ULA97pJXR

  Reference here: https://learn.ml5js.org/#/reference/posenet
*/

let video,
    poseNet,
    poses,
    pose,
    skeleton;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
}

function gotPoses(results) {
  // `results` is an array (the model can detect more than one person!)
  if (results.length > 0) {
    poses = results;
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
  // Can you adapt this sketch so that it draws the landmarks for more
  // than one person? Instead of selecting the first detection with `poses[0]`,
  // you would need to work with the whole array of poses (`poses`), and for
  // each element of this array, call `drawPose`. 
  // (You wouldn't need to have two global variables `poses` and `pose`,
  // the `poses` array would be enough, and you could extract the `.pose`
  // and the `.skeleton` inside `drawPose`.)
}


function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  image(video, 0, 0);

  // Idea to explore:

  // Like in previous sketches, there is no obligation to display the video,
  // and you could for instance imagine a blank canvas where a few points
  // from the body are used to control the position, or other parameters
  // (the size?) of text that is displayed on the screen, like in the Bill T.
  // Jones work seen in class.

  if (pose) drawPose(); // We call our function to draw the keypoints
}

// Dan Shiffman's function drawing various kinds of dots in various keypoints
function drawPose() {
  let eyeR = pose.rightEye; // Note how we can access various keypoints like properties
  let eyeL = pose.leftEye;

  // Note how Shiffman uses the distance between the eye to
  // adapt the size of the big red nose!
  let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
  fill(255, 0, 0);
  ellipse(pose.nose.x, pose.nose.y, d);
  fill(0, 0, 255);
  ellipse(pose.rightWrist.x, pose.rightWrist.y, 32);
  ellipse(pose.leftWrist.x, pose.leftWrist.y, 32);

  // Idea to explore:

  // You could instead use the distance between, say, the nose
  // and one wrist, that would allow a user to control the size of a
  // figure, or to make something happen if they touch their noses.

  // draw keypoints
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    fill(0,255,0);
    ellipse(x,y,16,16);
  }

  // draw bones, i.e. lines between keypoints
  for (let i = 0; i < skeleton.length; i++) {
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    strokeWeight(2);
    stroke(255);
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

// A click of the mouse logs the pose to the console, have a look at the object!
function mousePressed() {
  if (poses) {
    console.log('all poses:');
    console.log(poses);
    console.log('current pose:');
    console.log(pose);
    console.log('skeleton:');
    console.log(skeleton);
    console.log('-----------------------------');
  } else {
    console.log("No pose detected yet");
  }
}
