(() => {
    const root = document.documentElement;
    const themeToggle = document.querySelector("#theme-toggle");
    const storedTheme = localStorage.getItem("credit-risk-theme");

    if (storedTheme) {
        root.dataset.theme = storedTheme;
    }

    themeToggle?.addEventListener("click", () => {
        const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
        root.dataset.theme = nextTheme;
        localStorage.setItem("credit-risk-theme", nextTheme);
    });

    const defaultsNode = document.querySelector("#credit-risk-defaults");
    const defaults = defaultsNode ? JSON.parse(defaultsNode.textContent) : {};
    const form = document.querySelector("#risk-form");

    document.querySelector("#reset-form")?.addEventListener("click", () => {
        if (!form) return;

        Object.entries(defaults).forEach(([name, value]) => {
            const field = form.elements[name];
            if (field) {
                field.value = value;
            }
        });
    });

    const incomeInput = form?.elements.person_income;
    const amountInput = form?.elements.loan_amnt;
    const ratioInput = form?.elements.loan_percent_income;

    const updateRatio = () => {
        if (!incomeInput || !amountInput || !ratioInput) return;

        const income = Number(incomeInput.value);
        const amount = Number(amountInput.value);
        if (!Number.isFinite(income) || !Number.isFinite(amount) || income <= 0) return;

        const ratio = Math.min(amount / income, 1);
        ratioInput.dataset.suggested = ratio.toFixed(2);
    };

    incomeInput?.addEventListener("input", updateRatio);
    amountInput?.addEventListener("input", updateRatio);
    updateRatio();

    const modal = document.querySelector("#image-modal");
    const modalImage = document.querySelector("#modal-image");
    const modalTitle = document.querySelector("#modal-title");
    const closeModal = () => {
        modal?.classList.remove("is-open");
        modal?.setAttribute("aria-hidden", "true");
        if (modalImage) {
            modalImage.removeAttribute("src");
        }
    };

    document.querySelectorAll(".report-preview").forEach((button) => {
        button.addEventListener("click", () => {
            const image = button.dataset.image;
            const title = button.dataset.title || "";
            if (!image || !modal || !modalImage || !modalTitle) return;

            modalImage.src = image;
            modalImage.alt = title;
            modalTitle.textContent = title;
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
        });
    });

    document.querySelector(".modal-close")?.addEventListener("click", closeModal);
    modal?.addEventListener("click", (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeModal();
        }
    });
})();
