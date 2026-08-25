# PriceOye RAG System

> **An Intelligent, Anti-Hallucination Customer Support & Real-Time Product Retrieval Bot**  
> **Developed by:** Umer Hussain  
> **Live Application:** [priceoyerag.streamlit.app](https://priceoyerag.streamlit.app)

---

## 📸 System Overview

<div align="center">
  <img src="https://i.postimg.cc/hjVjHfdj/Screenshot-2026-08-24-21-14-43.png" alt="PriceOye RAG System Interface" width="85%" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 15px 0;">
</div>

---

## 🎯 Abstract & Problem Statement

Building an enterprise-grade customer support chatbot for **PriceOye**, one of Pakistan's leading e-commerce platforms. 

### Key Challenges Solved:
* **Strict Policy Enforcement:** Accurately navigating complex store rules (e.g., *3-day product replacement window* vs. *15-day bank refund processing*, *Leopards Courier* drop-off protocols, unboxing video verification for transit damage).
* **Real-Time Inventory Retrieval:** Fetching dynamic mobile prices, specs, and live product image renders from active databases.
* **Minimal-Hallucination Guardrails:** Minimizing critical retail risks such as phantom discount codes, false refund timelines, or incorrect inventory pricing.

---

## 💡 Mission & Value Proposition

In e-commerce, **LLM hallucinations directly translate to financial loss and brand erosion**. If an AI agent falsely promises a 50% discount or misleads a user on return validity, customer trust is permanently damaged.

This project demonstrates a production-ready, retrieval-augmented generation (RAG) architecture built with cutting-edge AI infrastructure (**Groq, Pinecone, Firecrawl, OpenRouter**). It delivers **verifiable responses, strict policy compliance, and real-time structured product cards**.

---

## 🛠️ Architecture & Tech Stack

| Tool / Technology | Role & Responsibility | Interface / Logo |
| :--- | :--- | :---: |
| **PriceOye** | Primary Data Source ([priceoye.pk](https://priceoye.pk)) | <img src="https://i.postimg.cc/ZqDxsDhd/priceoye-logo-2.png" width="40" alt="PriceOye"> |
| **Firecrawl** | Web Scraping, Structured Extraction & Crawling | <img src="https://i.postimg.cc/gj3yJz6Y/firecrawl-logo.png" width="40" alt="Firecrawl"> |
| **Pinecone** | High-Performance Vector Database | <img src="https://i.postimg.cc/mkWMKvp7/pinecone-logo.png" width="40" alt="Pinecone"> |
| **OpenRouter** | High-Dimensional Embedding Model API | <img src="https://i.postimg.cc/02cMCKvW/openrouter-logo.jpg" width="40" alt="OpenRouter"> |
| **Groq** | Ultra-Fast LLM Inference Engine | <img src="https://i.postimg.cc/Y9YLNX2Y/groq-logo.png" width="40" alt="Groq"> |

*Note: Google AI Studio and Supabase were additionally utilized during sandbox testing.*

---

## 🔄 End-to-End Workflow

<div align="center">

### Step 1: Data Extraction & Data Cleaning
[![Step 1](https://iili.io/CDYzfnf.md.png)](https://freeimage.host/i/CDYzfnf)

⬇️

### Step 2: Ingestion Pipeline
[![Step 2](https://iili.io/CDYzbDu.md.png)](https://freeimage.host/i/CDYzbDu)

⬇️

### Step 3: Retrieval and Response Pipeline
[![Step 3](https://iili.io/CDYIUdB.md.png)](https://freeimage.host/i/CDYIUdB)

</div>

---

## ✨ Expected System Capabilities
- 📱 **Interactive Product Cards:** Displays real-time pricing, key specifications, and direct product image renders inline.
- 📜 **Policy Precision:** Distinguishes nuances between replacement timelines, payment gateways, and shipping carrier constraints.
- 🎥 **Claim Guidance:** Step-by-step text workflows for submitting mandatory unboxing videos for transit damage disputes.
- 🛡️ **Grounded Responses:** Enforces strict context grounding to suppress hallucinatory claims.
