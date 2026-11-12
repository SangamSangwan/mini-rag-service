-- Most asked questions
SELECT question, COUNT(*) AS times_asked
FROM qa_log
GROUP BY question
ORDER BY times_asked DESC
LIMIT 10;

-- Average latency: cache hit vs miss
SELECT cached, COUNT(*) AS requests, ROUND(AVG(latency_ms)::numeric, 1) AS avg_latency_ms
FROM qa_log
GROUP BY cached;

-- Slowest questions
SELECT question, latency_ms, created_at
FROM qa_log
WHERE cached = FALSE
ORDER BY latency_ms DESC
LIMIT 5;
