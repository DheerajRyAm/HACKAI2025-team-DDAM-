"use client";
import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";

export default function Home() {
  const videoRef = useRef(null);
  const distractingVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const [score, setScore] = useState(0);
  const requestRef = useRef(null);
  const lastTimeRef = useRef(0);
  const fps = 30;
  let emojis = {
    anger: "😡",
    happy: "😃",
    fear: "😱",
    disgust: "🤢",
    sad: "😢",
    suprise: "😲",
  };
  let currentEmotion = "";
  let dropSpeed = 30;
  let theQueue = []; // the queue thats not really a queue

  function getRandomEmoji() {
    const keys = Object.keys(emojis); // Get all emoji keys
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    return { emotionName: randomKey, emoji: emojis[randomKey] };
  }

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

  function update(deltaTime) {
    theQueue.forEach((queueItem, index) => {
      queueItem.yPos += dropSpeed * deltaTime;
      if (
        Math.round(queueItem.yPos) > 270 &&
        Math.round(queueItem.yPos) < 350
      ) {
        // this section runs when they get it in the allotted time
        console.log(currentEmotion);
        if (currentEmotion == queueItem.emotion) {
          theQueue = theQueue.filter((value) => {
            if (value.id != queueItem.id) {
              return value;
            }
          });
        }
      } else if (Math.round(queueItem.yPos) == 350) {
        // this section runs if they dont do it in time
        theQueue = theQueue.filter((value) => {
          if (value.id != queueItem.id) {
            return value;
          }
        });
      }
    });
  }

  function draw(ctx) {
    ctx.font = "90px serif";
    theQueue.forEach((queueItem) => {
      ctx.fillText(
        `${queueItem.emoji}`,
        window.innerWidth - window.innerWidth / 3,
        queueItem.yPos
      );
    });
  }

  useEffect(() => {
    const startVideo = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    };

    startVideo();
    addToQueueInterval();
  }, []);

  const sendFrameToBackend = async () => {
    // all this stuff here happens in its own time, and doest wait for the game loop
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    // Capture frame and draw on canvas
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    // this needs to be before everything, so the ai has less to decipher
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
        currentEmotion = `${data.emotion}`;
      } catch (error) {
        console.error("Error sending frame:", error);
      }
    }, "image/jpeg");
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

    context.fillStyle = "blue";
    context.font = "48px serif";
    context.fillText(`Score: ${score}`, 10, 50);
  };

  // Send frames every 100ms
  useEffect(() => {
    const gameLoop = setInterval(sendFrameToBackend, 1000 / fps);
    return () => clearInterval(gameLoop);
  }, []);

  function addToQueueInterval() {
    // Set a new random timeout between 2s (2000ms) and 10s (10000ms)
    const randomTime = Math.floor(Math.random() * (10000 - 2000 + 1)) + 2000;
    let nextEmoji = getRandomEmoji();
    theQueue.push({
      emotion: nextEmoji.emotionName,
      emoji: nextEmoji.emoji,
      yPos: 0,
      id: uuidv4(),
    });

    setTimeout(addToQueueInterval, randomTime);
  }

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
