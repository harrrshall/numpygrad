/* Sidebar scroll-spy, sidebar toggle, copy-as-markdown, print, toast feedback.
   No dependencies. Progressive enhancement: the page is fully readable without JS. */
(function () {
  "use strict";

  // --- toast: brief confirmation message for toolbar actions ---
  var toastEl = document.getElementById("toast");
  var toastTimer = null;
  function toast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add("toast-show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.remove("toast-show");
    }, 2200);
  }

  // --- sidebar starts open on desktop (summary hidden), collapsed on mobile ---
  var sidebar = document.getElementById("sidebar");
  if (sidebar && window.matchMedia("(max-width: 60rem)").matches) {
    sidebar.removeAttribute("open");
  }

  // --- scroll-spy: highlight the section currently in view in the sidebar ---
  var links = {};
  document.querySelectorAll('.toc a[href^="#"]').forEach(function (a) {
    links[a.getAttribute("href").slice(1)] = a;
  });
  var heads = document.querySelectorAll(".content h2[id], .content h3[id]");
  var active = null;

  if ("IntersectionObserver" in window && heads.length) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          var a = links[e.target.id];
          if (a && a !== active) {
            if (active) active.classList.remove("is-active");
            a.classList.add("is-active");
            active = a;
          }
        });
      },
      { rootMargin: "-12% 0px -78% 0px", threshold: 0 }
    );
    heads.forEach(function (h) {
      io.observe(h);
    });
  }

  // --- hide / show the sidebar: toolbar button + a button on the sidebar
  //     itself, kept in sync and persisted across visits ---
  var toggleBtn = document.getElementById("toggle-sidebar-btn");
  var sidebarHideBtn = document.getElementById("sidebar-hide-btn");
  function applyHidden(hidden, announce) {
    document.body.classList.toggle("sidebar-hidden", hidden);
    if (toggleBtn) {
      toggleBtn.textContent = hidden ? "Show contents" : "Hide contents";
      toggleBtn.setAttribute("aria-pressed", hidden ? "true" : "false");
    }
    if (announce) toast(hidden ? "Contents panel hidden" : "Contents panel shown");
  }
  function persistHidden(hidden) {
    try {
      localStorage.setItem("sidebarHidden", hidden ? "1" : "0");
    } catch (e) {}
  }
  var storedHidden = null;
  try {
    storedHidden = localStorage.getItem("sidebarHidden");
  } catch (e) {}
  if (storedHidden === "1") applyHidden(true, false);
  if (toggleBtn) {
    toggleBtn.addEventListener("click", function () {
      var hidden = !document.body.classList.contains("sidebar-hidden");
      applyHidden(hidden, true);
      persistHidden(hidden);
    });
  }
  if (sidebarHideBtn) {
    sidebarHideBtn.addEventListener("click", function () {
      applyHidden(true, true);
      persistHidden(true);
    });
  }

  // --- toolbar: copy the whole page as a Markdown file ---
  var copyMdBtn = document.getElementById("copy-md-btn");
  if (copyMdBtn) {
    copyMdBtn.addEventListener("click", function () {
      var el = document.getElementById("source-md");
      var md = "";
      try {
        md = el ? JSON.parse(el.textContent) : "";
      } catch (e) {
        toast("Could not read the Markdown source");
        return;
      }
      if (!navigator.clipboard) {
        toast("Clipboard not available in this browser");
        return;
      }
      navigator.clipboard.writeText(md).then(
        function () {
          copyMdBtn.textContent = "Copied ✓";
          toast("Full Markdown copied to clipboard (" + Math.round(md.length / 1024) + " KB)");
          setTimeout(function () {
            copyMdBtn.textContent = "Copy as Markdown";
          }, 2000);
        },
        function () {
          toast("Copy failed — check browser permissions");
        }
      );
    });
  }

  // --- toolbar: print as book ---
  var printBtn = document.getElementById("print-btn");
  if (printBtn) {
    printBtn.addEventListener("click", function () {
      toast("Opening the print dialog — choose “Save as PDF” for a book");
      setTimeout(function () {
        window.print();
      }, 350);
    });
  }

  // --- copy buttons on code blocks ---
  if (navigator.clipboard) {
    document.querySelectorAll(".content pre, .getting-started pre").forEach(
      function (pre) {
        var btn = document.createElement("button");
        btn.className = "copy-btn";
        btn.type = "button";
        btn.textContent = "copy";
        btn.addEventListener("click", function () {
          var code = pre.querySelector("code");
          var text = code ? code.innerText : pre.innerText;
          navigator.clipboard.writeText(text).then(function () {
            btn.textContent = "copied ✓";
            setTimeout(function () {
              btn.textContent = "copy";
            }, 1500);
          });
        });
        pre.appendChild(btn);
      }
    );
  }
})();
