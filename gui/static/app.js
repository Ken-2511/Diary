/**
 * AI 日记助手 - 前端逻辑
 */

// ===== 全局状态 =====
let allDiaries = [];
let currentFilter = 'all';
let currentDiaryDate = null;
let isSearchMode = false;

// ===== DOM 元素 =====
const diaryList = document.getElementById('diary-list');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const clearSearchBtn = document.getElementById('clear-search-btn');
const reloadBtn = document.getElementById('reload-btn');
const filterBtns = document.querySelectorAll('.filter-btn');
const emptyState = document.getElementById('empty-state');
const diaryContent = document.getElementById('diary-content');
const statsEl = document.getElementById('stats');

// ===== API 调用 =====
async function fetchDiaries() {
    const res = await fetch('/api/diaries');
    return await res.json();
}

async function fetchDiary(date) {
    const res = await fetch(`/api/diary/${date}`);
    return await res.json();
}

async function searchDiaries(query) {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    return await res.json();
}

async function fetchStats() {
    const res = await fetch('/api/stats');
    return await res.json();
}

async function reloadDiariesCache() {
    const res = await fetch('/api/reload', { method: 'POST' });
    return await res.json();
}

// ===== 渲染函数 =====
function renderDiaryList(diaries) {
    diaryList.innerHTML = '';
    
    if (diaries.length === 0) {
        diaryList.innerHTML = '<li class="diary-item"><em>没有找到日记</em></li>';
        return;
    }
    
    diaries.forEach(diary => {
        const li = document.createElement('li');
        li.className = 'diary-item' + (diary.date === currentDiaryDate ? ' active' : '');
        li.dataset.date = diary.date;
        
        const statusIcon = diary.has_comment 
            ? '<span class="status-icon completed">🟢</span>'
            : '<span class="status-icon pending">🟡</span>';
        
        li.innerHTML = `
            <div class="diary-item-date">${diary.date_display}</div>
            <div class="diary-item-title">
                ${statusIcon}
                <span class="diary-title-text">${escapeHtml(diary.title)}</span>
            </div>
            ${diary.preview ? `<div class="diary-item-preview">${escapeHtml(diary.preview)}</div>` : ''}
        `;
        
        li.addEventListener('click', () => selectDiary(diary.date));
        diaryList.appendChild(li);
    });
}

function renderDiaryContent(diary) {
    // 隐藏空状态，显示内容
    emptyState.classList.add('hidden');
    diaryContent.classList.remove('hidden');
    
    // 填充内容
    document.getElementById('diary-title').textContent = diary.title;
    document.getElementById('diary-date').textContent = `📅 ${diary.date_display}`;
    document.getElementById('diary-word-count').textContent = `📝 ${diary.word_count} 字`;
    document.getElementById('diary-status').textContent = diary.has_comment ? '✅ 已评论' : '⏳ 待评论';
    
    // 渲染日记内容（Markdown）
    document.getElementById('diary-body').innerHTML = marked.parse(diary.diary || '*（无内容）*');
    
    // 渲染评论
    const commentSection = document.getElementById('comment-section');
    const commentBody = document.getElementById('comment-body');
    if (diary.comment) {
        commentSection.classList.remove('hidden');
        commentBody.innerHTML = marked.parse(diary.comment);
    } else {
        commentSection.classList.remove('hidden');
        commentBody.innerHTML = '<p class="no-comment">暂无 AI 评论</p>';
    }
    
    // 渲染 Token 使用
    const usageSection = document.getElementById('usage-section');
    const usageBody = document.getElementById('usage-body');
    if (diary.usage) {
        usageSection.classList.remove('hidden');
        usageBody.innerHTML = `
            <span>输入: <strong>${diary.usage.prompt_tokens}</strong> tokens</span>
            <span>输出: <strong>${diary.usage.completion_tokens}</strong> tokens</span>
            <span>总计: <strong>${diary.usage.total_tokens}</strong> tokens</span>
        `;
    } else {
        usageSection.classList.add('hidden');
    }
}

function renderStats(stats) {
    statsEl.innerHTML = `
        <span>📚 共 ${stats.total} 篇日记</span>
        <span>📅 本月 ${stats.this_month} 篇</span>
        <span>✅ 已评论 ${stats.with_comment} 篇</span>
        <span>⏳ 待评论 ${stats.without_comment} 篇</span>
        <span>📝 总字数 ${stats.total_words.toLocaleString()}</span>
    `;
}

// ===== 事件处理 =====
async function selectDiary(date) {
    // 如果点击的是已选中的日记，则关闭它
    if (currentDiaryDate === date) {
        currentDiaryDate = null;
        
        // 移除列表中的 active 状态
        document.querySelectorAll('.diary-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // 显示空状态，隐藏内容
        emptyState.classList.remove('hidden');
        diaryContent.classList.add('hidden');
        return;
    }
    
    currentDiaryDate = date;
    
    // 更新列表中的 active 状态
    document.querySelectorAll('.diary-item').forEach(item => {
        item.classList.toggle('active', item.dataset.date === date);
    });
    
    // 获取并显示日记详情
    const diary = await fetchDiary(date);
    renderDiaryContent(diary);
}

function applyFilter(filter) {
    currentFilter = filter;
    
    // 更新按钮状态
    filterBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });
    
    // 过滤日记
    let filtered = allDiaries;
    if (filter === 'pending') {
        filtered = allDiaries.filter(d => !d.has_comment);
    }
    
    renderDiaryList(filtered);
}

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    
    isSearchMode = true;
    clearSearchBtn.classList.remove('hidden');
    
    const results = await searchDiaries(query);
    renderDiaryList(results);
}

function clearSearch() {
    isSearchMode = false;
    searchInput.value = '';
    clearSearchBtn.classList.add('hidden');
    applyFilter(currentFilter);
}

async function reload() {
    reloadBtn.textContent = '⏳';
    reloadBtn.disabled = true;
    
    await reloadDiariesCache();
    await init();
    
    reloadBtn.textContent = '🔄';
    reloadBtn.disabled = false;
}

// ===== 工具函数 =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== 事件绑定 =====
searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});
clearSearchBtn.addEventListener('click', clearSearch);
reloadBtn.addEventListener('click', reload);

filterBtns.forEach(btn => {
    btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
});

// ===== 初始化 =====
async function init() {
    // 加载日记列表
    allDiaries = await fetchDiaries();
    renderDiaryList(allDiaries);
    
    // 加载统计
    const stats = await fetchStats();
    renderStats(stats);
    
    // 不自动选中日记，保持空白页面
}

// 启动应用
init();

// ===== 心跳机制 =====
// 每2秒发送心跳，保持服务器运行
setInterval(() => {
    fetch('/api/heartbeat').catch(() => {});
}, 2000);

// 页面关闭前不再发送心跳（服务器会在超时后自动退出）
window.addEventListener('beforeunload', () => {
    // 可以选择立即通知服务器退出（可选）
});
