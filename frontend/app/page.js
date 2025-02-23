"use client";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef(null);
  const distractingVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const [gesture, setGesture] = useState("Waiting for move...");
  const [score, setScore] = useState(10);
  const [highscore, setHighscore] = useState(10);
  const fps = 30;
  let emojis = {
    anger: "😡",
    happy: "😃",
    fear: "😱",
    disgust: "🤢",
    sad: "😢",
    suprise: "😲",
    neutral: "😐",
  };
  let start = Math.round(Date.now()) / 1000;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Set canvas size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const gameLoop = (timeStamp) => {
      const deltaTime = (timeStamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timeStamp;

      update(deltaTime);
      draw(ctx);

      requestRef.current = requestAnimationFrame(gameLoop);
    };

    requestRef.current = requestAnimationFrame(gameLoop);

    return () => cancelAnimationFrame(requestRef.current);
  }, []);

  function update(deltaTime) {}

  function draw(ctx) {
    const canvas = ctx.canvas;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const b = ball.current;
    ctx.fillStyle = "blue";
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
    ctx.fill();
  }

  let theQueue = [{ emoji: emojis.anger, startTime: 0 }];

  console.log(Math.floor(Date.now() / 1000));

  useEffect(() => {
    const startVideo = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    };

    startVideo();
  }, []);

  const sendFrameToBackend = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    // Capture frame and draw on canvas
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    context.font = "90px serif";
    context.fillText(
      `${theQueue[0].emoji}`,
      window.innerWidth - window.innerWidth / 3,
      0 + Math.floor(Date.now() / 1000)
    );
    // draw needed subway surfers gameplay
    context.drawImage(
      distractingVideoRef.current,
      0,
      canvas.height - 300,
      120,
      300
    );
    // draw line
    context.fillStyle = "red";
    context.fillRect(
      window.innerWidth - window.innerWidth / 3,
      window.innerHeight / 3,
      100,
      10
    );
    context.fillRect(
      window.innerWidth - window.innerWidth / 5,
      window.innerHeight / 3,
      100,
      10
    );

    context.fillStyle = "blue";
    context.font = "48px serif";
    context.fillText(`Score: ${score}`, 10, 50);

    context.fillStyle = "green";
    context.font = "48px serif";
    context.fillText(`Highscore: ${highscore}`, 10, 90);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append("frame", blob);

      try {
        const response = await fetch("http://localhost:8000/process_frame", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        setGesture(`Move: ${data.gesture}`);
      } catch (error) {
        console.error("Error sending frame:", error);
      }
    }, "image/jpeg");
  };

  // Send frames every 100ms
  useEffect(() => {
    const gameLoop = setInterval(sendFrameToBackend, 1000 / fps);
    return () => clearInterval(gameLoop);
  }, []);

  return (
    <div>
      <div className="flex place-content-center">
        <video
          ref={distractingVideoRef}
          src="/erm.mp4"
          autoPlay
          playsInline
          width="120"
          height="100"
          hidden
        ></video>
        <canvas
          ref={canvasRef}
          width={window.innerWidth}
          height={window.innerHeight}
        ></canvas>
        <video hidden ref={videoRef} autoPlay playsInline width="1000"></video>
      </div>
    </div>
  );
}
