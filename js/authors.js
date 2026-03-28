const AUTHOR_URL = {
    "xinxi-zhang": "https://scholar.google.com/citations?user=08KI_O0AAAAJ&hl=en",
    "song-wen": "https://scholar.google.com/citations?user=yTQbz6AAAAAJ&hl=en",
    "ligong-han": "https://scholar.google.com/citations?user=n2v43R4AAAAJ&hl=en",
    "felix-juefei-xu": "https://scholar.google.com/citations?user=dgN8vtwAAAAJ&hl=en",
    "akash-srivastava": "https://scholar.google.com/citations?user=2h6SZeEAAAAJ&hl=en",
    "junzhou-huang": "https://scholar.google.com/citations?user=X7KrguAAAAAJ&hl=en",
    "hao-wang": "https://scholar.google.com/citations?user=NrOA9QoAAAAJ&hl=en",
    "molei-tao": "https://example.com/molei-tao",
    "vladimir-pavlovic": "https://scholar.google.com/citations?user=8MQT8skAAAAJ",
    "dimitris-metaxas": "https://scholar.google.com/citations?user=a7VNhCIAAAAJ&hl=en",
    "tunyu-zhang": "https://scholar.google.com/citations?user=y3st15YAAAAJ&hl=en",
    "haizhou-shi": "https://scholar.google.com/citations?user=-OF9RG8AAAAJ&hl=en",
    "xiaoxiao-he": "https://scholar.google.com/citations?user=WmK6IOEAAAAJ&hl=en",
    "zhuowei-li": "https://scholar.google.com/citations?user=51OJEPcAAAAJ&hl=en",
    "kai-xu": "https://scholar.google.com/citations?user=kf3C60wAAAAJ&hl=en",
    "hao-wang2": "https://scholar.google.com/citations?user=A3WtYhAAAAAJ&hl=en",
    "shiwei-tan": "https://scholar.google.com/citations?user=XUsD3_kAAAAJ&hl=zh-CN",
    "quang-nguyen": "https://scholar.google.com/citations?user=SUuo7U4AAAAJ&hl=en",
    "quan-dao": "https://scholar.google.com/citations?user=g0RS3_kAAAAJ&hl=en",
    "alen-mrdovic": "https://www.linkedin.com/in/alen-mrdovic-615b881b6/",
    "chengzhi-mao": "https://scholar.google.com/citations?user=pTTEiHUAAAAJ&hl=en",
  };
  
  // linkify any <p class="project-authors"> that contains plain names
  const ME_ID = "xinxi-zhang";

  document.querySelectorAll(".project-authors[data-authors]").forEach((p) => {
  const ids = (p.dataset.authors || "").split(",").map(s => s.trim()).filter(Boolean);

  const starSet = new Set(
    (p.dataset.stars || "")
      .split(",")
      .map(s => s.trim())
      .filter(Boolean)
  );

  const pieces = ids.map((id) => {
    const url = AUTHOR_URL[id];
    const label = id.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    const star = starSet.has(id) ? "*" : "";

    const cls = ["author-link"];
    if (id === ME_ID) cls.push("is-me");

    return url
      ? `<a class="${cls.join(" ")}" href="${url}" target="_blank" rel="noopener">${label}${star}</a>`
      : `<span class="${cls.join(" ")}">${label}${star}</span>`;
  });

  p.innerHTML = pieces.join(", ");
});