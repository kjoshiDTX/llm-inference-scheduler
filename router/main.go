// Weeks 12-13: the router. A streaming reverse proxy over mini-engine
// workers with pluggable placement policies.
//
// Cancellation propagates for free: the upstream request uses the client
// request's context, so a client disconnect cancels the worker call, whose
// SSE generator teardown frees the KV blocks (server/worker.py).
//
// Usage:
//
//	go run . -listen :8000 -policy cache_aware -alpha 1 -beta 200 \
//	    -workers http://localhost:8001,http://localhost:8002
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages       []chatMessage `json:"messages"`
	ConversationID string        `json:"conversation_id"`
}

// prefixKey builds the canonical text whose prefix approximates the token
// prefix the worker will see (mirrors the chat template's message order).
func prefixKey(msgs []chatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|")
		b.WriteString(m.Role)
		b.WriteString("|>")
		b.WriteString(m.Content)
	}
	return b.String()
}

type Router struct {
	workers []*Worker
	policy  Policy
	client  *http.Client
	total   atomic.Int64
}

func (rt *Router) pickAndCount(body []byte) (*Worker, string) {
	var req chatRequest
	_ = json.Unmarshal(body, &req)
	key := prefixKey(req.Messages)
	convID := req.ConversationID
	if convID == "" && len(req.Messages) > 0 {
		// Fall back: first user message identifies the conversation.
		for _, m := range req.Messages {
			if m.Role == "user" {
				convID = m.Content
				break
			}
		}
	}
	w := rt.policy.Pick(rt.workers, key, convID)
	// Record what this worker is about to cache: the full request prefix.
	// (The response tokens extend it, but message-boundary granularity is
	// enough for multi-turn reuse: the next turn resends this prefix.)
	w.Tree.Insert(key)
	return w, key
}

func (rt *Router) handleChat(wr http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(wr, err.Error(), http.StatusBadRequest)
		return
	}
	worker, _ := rt.pickAndCount(body)
	worker.Inflight.Add(1)
	worker.Requests.Add(1)
	rt.total.Add(1)
	defer worker.Inflight.Add(-1)

	up, err := http.NewRequestWithContext(
		req.Context(), "POST", worker.BaseURL+"/v1/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		http.Error(wr, err.Error(), http.StatusInternalServerError)
		return
	}
	up.Header.Set("Content-Type", "application/json")

	resp, err := rt.client.Do(up)
	if err != nil {
		worker.Healthy.Store(false)
		http.Error(wr, fmt.Sprintf("worker %s unreachable: %v", worker.ID, err),
			http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	wr.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	wr.Header().Set("X-Routed-To", worker.ID)
	wr.WriteHeader(resp.StatusCode)

	flusher, canFlush := wr.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := wr.Write(buf[:n]); werr != nil {
				return // client gone; context cancellation stops the worker
			}
			if canFlush {
				flusher.Flush() // SSE: every chunk out the door immediately
			}
		}
		if rerr != nil {
			return
		}
	}
}

func (rt *Router) handleMetrics(wr http.ResponseWriter, _ *http.Request) {
	wr.Header().Set("Content-Type", "application/json")
	type wm struct {
		ID       string `json:"id"`
		Requests int64  `json:"requests"`
		Inflight int64  `json:"inflight"`
		Healthy  bool   `json:"healthy"`
	}
	out := struct {
		Policy  string `json:"policy"`
		Total   int64  `json:"total_requests"`
		Workers []wm   `json:"workers"`
	}{Policy: rt.policy.Name(), Total: rt.total.Load()}
	for _, w := range rt.workers {
		out.Workers = append(out.Workers, wm{
			ID: w.ID, Requests: w.Requests.Load(),
			Inflight: w.Inflight.Load(), Healthy: w.Healthy.Load(),
		})
	}
	_ = json.NewEncoder(wr).Encode(out)
}

func (rt *Router) healthLoop() {
	for {
		for _, w := range rt.workers {
			resp, err := rt.client.Get(w.BaseURL + "/health")
			ok := err == nil && resp.StatusCode == 200
			if resp != nil {
				resp.Body.Close()
			}
			was := w.Healthy.Swap(ok)
			if was != ok {
				log.Printf("worker %s healthy=%v", w.ID, ok)
			}
		}
		time.Sleep(2 * time.Second)
	}
}

func main() {
	listen := flag.String("listen", ":8000", "listen address")
	workersFlag := flag.String("workers", "http://localhost:8001", "comma-separated worker URLs")
	policyFlag := flag.String("policy", "cache_aware", "round_robin | hash | cache_aware")
	alpha := flag.Float64("alpha", 1.0, "cache_aware: weight of prefix overlap (chars)")
	beta := flag.Float64("beta", 1000.0, "cache_aware: penalty per in-flight request")
	flag.Parse()

	var workers []*Worker
	for i, u := range strings.Split(*workersFlag, ",") {
		w := &Worker{
			ID:      fmt.Sprintf("w%d", i+1),
			BaseURL: strings.TrimRight(strings.TrimSpace(u), "/"),
			Tree:    NewPrefixTree(),
		}
		w.Healthy.Store(true)
		workers = append(workers, w)
	}

	var policy Policy
	switch *policyFlag {
	case "round_robin":
		policy = &RoundRobin{}
	case "hash":
		policy = &HashAffinity{}
	case "cache_aware":
		policy = &CacheAware{Alpha: *alpha, Beta: *beta}
	default:
		log.Fatalf("unknown policy %q", *policyFlag)
	}

	rt := &Router{
		workers: workers,
		policy:  policy,
		client:  &http.Client{}, // no timeout: streams are long-lived
	}
	go rt.healthLoop()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", rt.handleChat)
	mux.HandleFunc("/router/metrics", rt.handleMetrics)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	log.Printf("router listening on %s, policy=%s, workers=%d",
		*listen, policy.Name(), len(workers))
	log.Fatal(http.ListenAndServe(*listen, mux))
}
