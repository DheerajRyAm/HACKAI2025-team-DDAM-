"use client";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [gesture, setGesture] = useState("Waiting for move...");

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
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
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

  // Send frames every 500ms
  useEffect(() => {
    const interval = setInterval(sendFrameToBackend, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Dance Game 🎵💃🕺</h1>
      <video ref={videoRef} autoPlay playsInline width="500"></video>
      <canvas
        ref={canvasRef}
        width="224"
        height="224"
        style={{ display: "none" }}
      ></canvas>
      <h2 style={{ marginTop: "20px", color: "blue" }}>{gesture}</h2>
    </div>
  );
}
