---
title: LLM Finetune 梳理
date: 2023-12-13T22:14:17+08:00
draft: true
categories: [理论]
tags: [AI, LLM]
math: mathjax
---

> 最近比较研究了一些 LLM Fine-tune 的细节，整理一下，方便以后查阅。

## 0. 背景

LLM 模型在 NLP 领域基本已经成为了标配，但是要 LLM 在实际业务中落地，往往需要修改和适配才能符合业务具体要求：比如QA系统对回答精准度的要求；角色扮演游戏对语气和一致性的要求等等。

LLM 可以进行适配或者 Finetune 是基于一个假设：LLM 本身已经蕴含了我们需要的知识和能力，只是需要我们用一定的方式来引导它。而这种引导的方式，都可以叫 Finetune。

基于 LLM 的结构特点和性质，在 Finetune 上有很多已有的尝试，包括：

- 全模型参数 Finetune
- Prompt Finetune
- 部分参数 Finetune (PEFT)

本文也将从这三个方面来梳理 LLM Finetune 的细节。

## 1. LLM Finetune 的基本类型

### 全模型参数 Finetune


### Prompt Finetune

说白了就是普通人在使用 LLM 的时候最常用的方式——挑选一个合适的 Prompt 来引导 LLM 生成想要的结果。通常我们都会尝试使用多个不同的 Prompt，然后挑选最好的一个。比如一些神奇的“咒语”一样的 Prompt 可以让 LLM 生成一些很有意思的结果。

这种方式的优点是简单直接，缺点是需要人工设计 Prompt，而且很难保证 Prompt 的效果。

## 2. 各种 Finetune 的优劣对比

## 3. Finetune 的能力边界和业务场景

## 5. 总结
