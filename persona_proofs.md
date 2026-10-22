# Mamba-1.4B Multi-Persona Routing & Quantization Validation

This document provides side-by-side evidence of domain adaptation and verifies that Layer 2 Int8 quantization does not degrade reasoning quality.

## Base Model
**Prompt:** Explain the concept of dynamic equilibrium in a basic system.

### Unquantized (FP32)
> Explain the concept of dynamic equilibrium in a basic system.
The term "dynamic" is used to describe an object that changes with time, such as temperature or pressure. The word also describes something undergoing change at any given moment; for example, when you turn on your light switch and it illuminates all night long (the brightness varies from room-to-room). A static state does not vary over space nor do objects have fixed states during their lives: they are always changing. For this reason we say things like "a lamp burns out," meaning its intensity decreases until eventually no more energy can be produced by burning fuel. This means there was never enough available to produce intense illumination throughout the life span of the bulb. In contrast, if someone were trying to make fireflies glow brighter so people could

*Latency: 20.49 seconds*

### Quantized (Int8)
> Explain the concept of dynamic equilibrium in a basic system.
The concepts used to describe systems are often based on their behavior, or what they do when you observe them and measure certain properties such as mass changes (mass is conserved). In this case we say that these objects have an internal state which can be described by its position vector $\vec{x}$ at any time $t$ along with some other variables describing it’s motion like velocity ($\dot{\vec x}$) etc.. This description however does not take into account how those positions change over space-time intervals; therefore if one were able to find out exactly where each object was located throughout history then he would also know about all possible states the system could occupy during said interval since there will always exist two different configurations for every point within his domain

*Latency: 31.63 seconds*

---

## Software Engineer
**Prompt:** Write a C++ function to reverse a singly linked list and explain its time complexity.

### Unquantized (FP32)
> A doubly-linked node in the singly-linked list can be reversed by traversing through it from head to tail, starting with the first element of the list and then reversing each link as we traverse downwards until reaching the last element. This process is called "preorder" traversal because all nodes are visited before they reach the root (head) or end point (tail). The preorder traversal takes about n/2 steps for every step of one iteration, where n represents number of elements in the singly-linked list.

*Latency: 18.79 seconds*

### Quantized (Int8)
> The time complexity of the algorithm is O(n). The reason for this is that each node in the tree will be visited at most n times, since it must be traversed from head to tail until all nodes are reached one by one.

*Latency: 9.18 seconds*

---

## Clinical Researcher
**Prompt:** What are the primary serum tumor markers used in the diagnosis of hepatocellular carcinoma?

### Unquantized (FP32)
> AFP (alpha-fetoprotein) and Hepatitis B surface antigen.

*Latency: 3.57 seconds*

### Quantized (Int8)
> The most common serum tumor marker for hepatocellular carcinoma is alpha-fetoprotein (AFP). AFP is a protein that is produced by many types of liver cells, including those found in the bile ducts and hepatocytes. Elevated levels of AFP have been associated with various stages of HCC, ranging from early to advanced disease. The level of AFP can be measured using blood tests or imaging studies such as ultrasound scans or CT scans. AFP has traditionally been considered an effective screening tool for detecting HCC at earlier stages when it may still be treatable, but other diagnostic methods like biopsy or surgery may also be necessary depending on the stage of the cancer and individual patient factors.

*Latency: 24.71 seconds*

---

## Legal Counsel
**Prompt:** Explain the scope of Article 21 of the Indian Constitution regarding the right to a clean environment.

### Unquantized (FP32)
> The scope of Article 21 of the Indian Constitution, which guarantees protection against violation of personal liberty and life with dignity, does not explicitly mention environmental issues or pollution prevention measures. However, it is important to note that the right to live peacefully and in harmony with nature has been recognized as an integral part of the right to livelihood under Articles 14 and 19. The Supreme Court has held that these rights are protected by Article 21 (see: Ashok Kumar Gupta & Ors vs State Of U.P & Ors, 1987 AIR 515, 1987 SCR (2) 488). This implies that individuals have a constitutional duty to protect their natural habitat and prevent any interference with their peaceful living conditions.

*Latency: 28.14 seconds*

### Quantized (Int8)
> The scope of Article 21 of the Indian Constitution, which guarantees certain fundamental rights and liberties for all citizens, does not directly address the issue of environmental protection or pollution control in India. However, it is important to note that the right to life and personal liberty guaranteed under Articles 14 and 19 also includes an implicit recognition of the need for social justice and equality. The right to live with dignity and respect for human dignity encompasses both physical and mental well-being. Therefore, while Article 21 may not explicitly mention the right to a healthy environment, its underlying principles can be extended to protect the quality of our natural resources and prevent their degradation due to industrial activities. This would include measures such as protecting biodiversity, preventing unsustainable land use practices, promoting sustainable agriculture

*Latency: 28.24 seconds*

---

