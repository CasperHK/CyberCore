# 🌌 CyberCore: The AI-Native Matrix OS

**為 AI 時代打造的「數據織網」作業系統**

CyberCore 摒棄傳統以 CPU 為中心的架構，將整個數據中心視為一台巨型電腦（**Datacenter-as-a-Computer**）。它以 **Rust** 的嚴格記憶體安全與所有權模型作為安全基礎，結合 **Mojo** 的異構運算效能，實現數據在網路、儲存與 GPU/NPU 之間的**零拷貝超導流動**。

CyberCore 不只是作業系統，更是**數據中心級的交換矩陣 (Exchange Fabric)**，讓「網路即總線」（Network-as-a-Bus），數據成為主動流動的資產，而非被動檔案。

## 💡 核心哲學：Data-Centric Architecture

傳統 OS 把數據當成「需要搬運的文件」，CyberCore 則徹底轉向**以數據為中心**的設計：

- **網路即總線**：數據透過 Rust 驅動的 RDMA/SmartNIC 協議，直接在網卡、儲存與 GPU 之間進行 Peer-to-Peer 傳輸，CPU 不再擔任中轉站。
- **簽名即安全 (Proof-of-Trust)**：CPU 從「搬運工」轉型為「公證人」。所有數據流必須攜帶由 Rust 微內核簽發的硬體級加密憑證（結合 TEE / IOMMU），實現 **Trust-Verify Data Path (TVDP)**。
- **計算下放**：部分預處理與邏輯可在網卡或儲存端由 Mojo 算子完成，進一步降低延遲。

這種設計大幅縮小攻擊面，同時實現極致的效能與安全隔離。

## 🚀 關鍵技術亮點

### 1. 🛡️ Trust-Verify Data Path (TVDP)
利用 Rust 的所有權規則與硬體 IOMMU 隔離，實現**零拷貝 P2P 傳輸**。微內核僅在控制平面進行快速權限驗證與憑證簽發，數據平面完全 bypass CPU 運算，兼顧安全與效能。

### 2. ⚡ Mojo Acceleration Layer
內嵌 Mojo 運算引擎，負責 AI 負載預測、動態資源調度與矩陣運算優化。讓作業系統本身具備「預測性」調度能力，在硬體層提前準備緩存與算力。

### 3. 🕸️ Distributed Fabric Kernel
將多機房的記憶體抽象為統一地址空間（Global Address Space）。開發者專注於計算邏輯，CyberCore 自動優化物理傳輸路徑，消除傳統機器邊界帶來的複雜度。

## 🛠 系統組成

- **CyberKernel (Rust)**  
  微內核 + 硬體抽象層，負責記憶體管理、capability 系統、TEE 簽名、IOMMU 映射與安全隔離。作為整個系統的「守衛」與「公證人」。

- **CyberFabric (Network Core)**  
  零拷貝網路協議棧，基於 RDMA、SmartNIC/DPU 與 GPUDirect 技術，實現「網路即總線」的數據高速流動。

- **Mojo Runtime (Compute Engine)**  
  高性能異構運算運行時，專注於 AI 算子、動態調度與邊緣預處理。

## 📅 開發路線圖 (Roadmap)

- **Phase 1: Secure Link** — 實作 Rust 微內核的硬體加密簽名機制與 Trust-Verify Data Path，確保 P2P 傳輸的安全性。
- **Phase 2: Mojo Engine Integration** — 完成 Rust 與 Mojo 的安全交互接口（FFI + 共享記憶體 + capability 傳遞），嵌入智能調度器。
- **Phase 3: Global Address Space** — 實現跨節點統一記憶體抽象，消除機器邊界。
- **Phase 4: AIOS Standard** — 釋出 CyberCore SDK，定義 AI 原生應用的開發規範與協議。

> **目前狀態**：早期原型開發階段，重點在 Rust 微內核與零拷貝網路核心。歡迎有經驗的貢獻者加入。

## 🤝 參與創造

我們正在尋找具備遠見與實力的共同創造者：

- **系統架構師**：對「去 CPU 中心化」與 Data-Centric 設計有熱情。
- **Rust 系統工程師**：微內核、驅動、IOMMU、capability 系統。
- **加密與硬體安全專家**：TEE、硬體簽名、Trust-Verify 機制。
- **Mojo / AI 效能工程師**：異構算子優化、預測性調度。
- **低階網路專家**：RDMA、SmartNIC/DPU、GPUDirect 經驗。

貢獻前請閱讀 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 📜 許可證

MIT License – 鼓勵開源協作與商業創新。

---

**"Stop managing files, start directing data."**

CyberCore 正在重新定義 AI 時代算力的物理邊界與安全邊界。

如果你對 **Rust + Mojo** 的異構 AI 原生系統、零拷貝網路核心，或 Datacenter-as-a-Computer 願景感興趣，歡迎一起打造未來。
