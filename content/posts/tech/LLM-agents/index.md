+++
title = 'LLM Agents'
date = 2023-12-14T10:32:36+08:00
categories = ['技术']
tags = ['AI', 'LLM', 'Agents']
math = 'mathjax'
draft = false
+++

## LLM NPC System

本文试图设计一个游戏中的 NPC 系统，目标是让游戏环境更加真实，丰满，有趣。为了达到这个目标，我们试图实现一个在开放世界场景下的大量 NPC 驱动系统。它具有以下特点：

- 大量的 NPC 数量
- 丰富的 NPC 种类
- 真实的 NPC 行为
- 有趣的 NPC 互动

我们希望引入 LLM 的能力来增强 NPC 驱动系统。一个完整的 NPC 驱动系统应该包括以下几个部分：

- 对话系统
- 行为系统
- 情感系统
- 记忆系统
- 社交系统

下面详细分析这些系统，以及 LLM 可以在这些系统中扮演的角色。

### 对话系统

语言 -> 玩家/其他 NPC

对话系统是一个 NPC 与玩家或者其他 NPC 交流的系统。这也是 NPC 最核心的能力，因为这是玩家与 NPC 互动时最容易感知的一个方面。相比于传统的对话树结构，LLM 可以自由生成对话，有更好的灵活性和可扩展性。目前研究最多的也是这方面的内容，这里暂时不做展开。

可能问题点：

在线直出的推理结果，可能会有很多合规问题，这点要格外注意。

### 行为系统

动作 -> 环境/其他 NPC

行为系统是一个 NPC 与环境交互的系统。传统的行为系统分为两方面：

- NPC 方面：行为树，状态机，效用函数
- 环境方面：物理引擎，碰撞检测，路径规划，交互逻辑

#### NPC 方面

LLM 可以考虑代替行为树，根据不同的观测值和人物设定来生成不同的行为。

> 参考 Voyager 项目中的实现，生成行为树的部分或全部。

可能问题点：

Prompt 和 Response 的格式规范问题

#### 环境方面

考虑抽象简化环境内容，模拟引擎接口，规定简化的环境交互接口和交互逻辑。需要显式地规定环境的交互逻辑，这样才能保证 LLM 生成的动作是合法的。

抽象内容：

- 物体
- 位置
- 人物
- 时间
- 动作

可能问题点：

- 规定合法的环境交互逻辑
- 思考 LLM 来代替环境交互逻辑的可能性（不够精确）
- 动作和交互的时序性保证

### 情感系统

维护一个情感状态机，根据环境和对话的内容来更新情感状态机的状态。LLM 可以用来驱动情感状态机的状态转移。

可能问题点：

情感系统的状态机设计，规则生成，即从人设映射到状态机的状态转移规则。

### 记忆系统

维护一个记忆库，记录 NPC 应该记忆的内容，包括：

- log 信息
  - 对话历史
  - 动作历史
  - 时空信息
- 背景信息
  - 人物设定
  - 社交关系
- 环境信息
  - 时空信息
  - 地图信息
  - 物体信息
- 知识信息

可以用向量数据库来存储这些信息，LLM 可以用来生成这些信息。

可能问题点：

向量数据库的检索设计。

### 社交系统

社交系统维护的是人（NPC）之间的关系情况。也是一个状态机，根据环境和对话的内容来更新状态机的状态。LLM 可以用来驱动状态机的状态转移。

可能问题点：

社交系统的状态机设计，规则生成，即从人设映射到状态机的状态转移规则。

## 为了减少成本的层级设计

LLM 的调用成本还是很高，用它来驱动上述 NPC 系统会有两方面的成本问题：

- 一方面同时在线的玩家时刻都在和 NPC 交互，每次交互都会调用若干次 LLM 推理
- 另一方面为了维持一个大量的 NPC 系统，需要同时在线大量的 NPC，这也会导致大量的 LLM 调用

因此考虑设计一个层级的 NPC 系统，根据和玩家的交互程度，将 NPC 分为三个层级：

- 和玩家直接交互的 NPC
- 玩家可以看到/听到的 NPC
- 玩家看不到/听不到的 NPC

对于玩家可以直接交互的 NPC，我们可以用 LLM 来驱动，因为这些 NPC 的交互次数是最多的，而且玩家可以直接感知到，所以这些 NPC 的交互质量要求也是最高的。

对于其他两种 NPC，我们可以用一种由粗到细的方式来驱动。

具体地，可以将所有其他 NPC 的行为先用一种粗粒度的方式来驱动，如传统行为树模拟或 LLM 生成的事件简报。然后根据玩家的感知程度，Lazy Update 这些 NPC 的具体行为或对话内容。

上述所有的生成内容都会落盘到数据库中，这样可以保证玩家离线后，NPC 的行为和对话内容不会发生变化。

To be continued...