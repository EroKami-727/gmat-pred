# How Our Dataset and Machine Learning Works

To make our "Early Exit" system work, we need a lot of data. Machine Learning is not magic; it is pattern recognition. This document explains what data we are collecting and how the ML uses it to predict the future.

## 1. What is the Data?
Our data is purely numbers. We don't feed images or 3D graphics to the AI. We feed it rows and columns of physics data. 

Since spaceflight happens over time, we use **Time-Series Data**. Every time GMAT moves the simulation forward (e.g., every 60 seconds of flight), we take a "snapshot" of the spacecraft.

**The Columns (Features) we collect for each snapshot:**
* Time elapsed
* Spacecraft Speed (Velocity X, Y, Z)
* Spacecraft Position (X, Y, Z coordinates)
* Target Position (e.g., Moon X, Y, Z) 
* Fuel remaining

*Important Note:* We do **not** use the "Date" or "Calendar Year" as a column. If we give the ML a date, it won't know where the planets are. Instead, we let GMAT calculate where the Moon/Target is, and we feed those exact X, Y, Z coordinates to the ML. This way, the ML always knows what the spacecraft is aiming at, no matter what year the mission takes place.

## 2. The Target Variable (What we want to predict)
At the end of every simulation, we record one final label:
* `1` = Success (Reached target)
* `0` = Failure (Crashed, ran out of fuel, or got lost in space)

In our dataset, every single "snapshot" of a flight is tagged with this final outcome. 

## 3. How the ML Actually Predicts (Not just repeating)
It is easy to think the ML just memorizes numbers and says, "I've seen these exact numbers before, so it will crash." But because our inputs are long decimals (like `Speed = 7.1234 km/s`), **the ML will never see the exact same simulation twice.**

Instead of memorizing, the ML learns the **Corridor of Survival**.
Imagine throwing a dart. The moment it leaves your hand, you can usually guess if it will hit the board or fall short, just by looking at its speed and angle. 

Our ML uses a special network called an **LSTM (Long Short-Term Memory)**. The LSTM is great at looking at how things change over time. 
* It looks at the first 2 hours of a 3-day flight. 
* It notices: "The speed is dropping a little too fast, and the angle is slightly off compared to where the Moon is right now."
* Even if the spacecraft hasn't crashed yet, the LSTM knows that this specific *trend* leads to a failure. 

It also understands trade-offs! It learns that a slightly lower speed is okay **if** the spacecraft is at a higher altitude. It accidentally learns the rules of physics just by looking at thousands of examples.

## 4. How We Force the AI to Learn
If we run 10,000 simulations and they all succeed, the ML will be useless. It will just guess "Success" every time. 

To make the AI smart, we have to show it what failure looks like. When we generate our random Monte Carlo inputs, we will deliberately use a wide variance (e.g., changing the payload mass by 20%, or tweaking launch angles randomly) so that **at least 30% to 40% of our training flights fail**. The AI needs to see thousands of crashes to learn how to predict them early.