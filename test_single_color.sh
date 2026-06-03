#!/bin/bash
(
  sleep 0.5
  echo "15"
  sleep 0.5
  echo "1"
  sleep 5
  echo "0"
  echo "0"
) | ./build/Lab1 2>&1 | grep -A 20 "3x3 Grid Colors" | head -50
