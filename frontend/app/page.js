"use client";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef(null);
  const distractingVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const [gesture, setGesture] = useState("Waiting for move...");
  const [score, setScore] = useState(10);
  const [highscore, setHighscore] = useState(10);
  const fps = 60;

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

    // Capture frame
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    context.drawImage(
      distractingVideoRef.current,
      0,
      canvas.height - 300,
      100,
      300
    );
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
    context.fillText(`Score: ${highscore}`, 10, 90);

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
    const interval = setInterval(sendFrameToBackend, 1000 / fps);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <div className="flex place-content-center">
        <video
          ref={distractingVideoRef}
          src="/erm.mp4"
          autoPlay
          playsInline
          width="100"
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
