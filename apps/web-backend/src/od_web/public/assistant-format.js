/* 以 DOM 文本节点渲染助手内容，不执行模型输出中的 HTML。 */
"use strict";

function renderAssistantText(target, content) {
  target.replaceChildren();
  const text = String(content ?? "").replace(/\r\n?/g, "\n");
  const inline = (parent, value) => {
    const pattern = /\*\*([^*\n]+)\*\*|__([^_\n]+)__|`([^`\n]+)`|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(?<!\*)\*([^*\n]+)\*(?!\*)/g;
    let start = 0, match;
    while ((match = pattern.exec(value))) {
      parent.append(document.createTextNode(value.slice(start, match.index)));
      const tag = match[1] || match[2] ? "strong" : match[3] ? "code" : match[4] ? "a" : "em";
      const element = document.createElement(tag);
      element.textContent = match[1] || match[2] || match[3] || match[4] || match[6];
      if (tag === "a") { element.href = match[5]; element.target = "_blank"; element.rel = "noopener noreferrer"; }
      parent.append(element); start = pattern.lastIndex;
    }
    parent.append(document.createTextNode(value.slice(start)));
  };
  let list = null, listType = "", paragraph = null;
  for (const line of text.split("\n")) {
    if (!line.trim()) { list = null; listType = ""; paragraph = null; continue; }
    const heading = line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/);
    const item = line.match(/^\s*(?:([-+*])\s+|(\d+)[.)]\s+)(.+)$/);
    if (heading) {
      list = null; listType = ""; paragraph = null;
      const block = document.createElement("h4"); inline(block, heading[1]); target.append(block);
    } else if (item) {
      paragraph = null; const type = item[2] ? "ol" : "ul";
      if (!list || listType !== type) { list = document.createElement(type); listType = type; target.append(list); }
      if (item[2] && list.childElementCount === 0) list.start = Number(item[2]);
      const block = document.createElement("li"); inline(block, item[3]); list.append(block);
    } else if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      list = null; listType = ""; paragraph = null; target.append(document.createElement("hr"));
    } else {
      list = null; listType = "";
      if (!paragraph) { paragraph = document.createElement("p"); target.append(paragraph); }
      else paragraph.append(document.createElement("br"));
      inline(paragraph, line.trim());
    }
  }
}
