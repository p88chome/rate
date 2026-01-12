import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from 'primereact/card';
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import { FileUpload } from 'primereact/fileupload';
import { Toast } from 'primereact/toast';
import { Sidebar } from 'primereact/sidebar';
import { ScrollPanel } from 'primereact/scrollpanel';
import { Avatar } from 'primereact/avatar';
import { Tag } from 'primereact/tag';

import 'primeicons/primeicons.css';
import 'primereact/resources/themes/lara-light-indigo/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeflex/primeflex.css';

// Add custom styles for markdown content
import './App.css';

//const API_BASE = 'https://rateagent.onrender.com';
const API_BASE = import.meta.env.VITE_API_URL || '/api';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '您好！我是稽核 AI 助手。我可以協助您進行各項稽核分析工作。請問您今天想要進行什麼樣的查核呢？' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [configStatus, setConfigStatus] = useState(null);
  const [sidebarVisible, setSidebarVisible] = useState(false);


  const toast = useRef(null);
  const messagesEndRef = useRef(null);

  // Initial check
  useEffect(() => {
    checkConfig();
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkConfig = async () => {
    try {
      const res = await axios.get(`${API_BASE}/config-status`);
      setConfigStatus(res.data);
      if (!res.data.configured) {
        toast.current.show({ severity: 'warn', summary: '設定遺失', detail: '請檢查後端 .env 檔案設定。' });
      }
    } catch (e) {
      console.error(e);
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      toast.current.show({ severity: 'success', summary: '上傳成功', detail: '檔案處理完成', life: 3000 });
      setMessages(prev => [...prev, { role: 'assistant', content: `**${file.name}** 已處理完畢！您可以開始針對此檔案進行提問。` }]);
      event.options.clear();
    } catch (error) {
      console.error(error);
      toast.current.show({ severity: 'error', summary: '錯誤', detail: error.response?.data?.detail || '上傳失敗', life: 5000 });
      event.options.clear();
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsLoading(true);

    // Show quick actions after first user message
    // Moved to after response to prevent buttons appearing before AI reply


    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: userMsg });

      let aiResponse = res.data.response;

      // If this was the first interaction (triggering quick actions), append guidance


      setMessages(prev => [...prev, {
        role: 'assistant',
        content: aiResponse,
        image: res.data.image,
        file: res.data.file
      }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'assistant', content: `發生錯誤: ${error.response?.data?.detail || error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const startItems = (
    <div className="flex align-items-center gap-2">
      <Button icon="pi pi-bars" text rounded onClick={() => setSidebarVisible(true)} className="text-gray-600 hover:text-black mr-2" />
      <i className="pi pi-chart-bar text-deloitte text-2xl"></i>
      <span className="font-bold text-xl text-900">稽核 AI 助手</span>
    </div>
  );

  const endItems = (
    <div className="flex align-items-center gap-2">
      {configStatus?.configured ? (
        <Tag severity="success" value="已連線" icon="pi pi-check" className="mr-2" style={{ backgroundColor: '#86BC25' }} />
      ) : (
        <Tag severity="danger" value="未連線" icon="pi pi-times" className="mr-2" />
      )}
    </div>
  );

  return (
    <div className="flex flex-column h-screen bg-gray-50 font-sans">
      <Toast ref={toast} />

      {/* Glassmorphism Header */}
      <div className="flex justify-content-between align-items-center p-3 z-5 relative backdrop-blur-md bg-white/80 border-bottom-1 border-gray-200 shadow-sm sticky top-0">
        {startItems}
        {endItems}
      </div>

      {/* Sidebar for Upload & Tools */}
      <Sidebar
        visible={sidebarVisible}
        onHide={() => setSidebarVisible(false)}
        position="left"
        className="w-full md:w-20rem shadow-3 border-round-right-2xl overflow-hidden p-0"
        showCloseIcon={false}
        content={({ closeIconRef, hide }) => (
          <div className="flex flex-column h-full bg-white/90 backdrop-blur-md">
            <div className="p-4 bg-deloitte-gradient text-white flex justify-content-between align-items-center">
              <span className="font-bold text-xl"><i className="pi pi-box mr-2"></i>工具箱</span>
              <Button icon="pi pi-angle-left" rounded text className="text-white hover:bg-white/20" onClick={hide} />
            </div>

            <div className="p-4 flex-1">
              <div className="mb-5">
                <h4 className="text-gray-700 mb-3 font-semibold">系統狀態</h4>
                <div className="flex flex-column gap-2">
                  <div className="flex align-items-center justify-content-between p-3 border-1 border-gray-200 border-round-lg bg-gray-50">
                    <span className="text-sm text-gray-600"><i className="pi pi-server mr-2 text-deloitte" />運算核心</span>
                    <Tag severity={configStatus?.configured ? "success" : "danger"} value={configStatus?.configured ? "運作中" : "離線"} className="text-xs" style={configStatus?.configured ? { backgroundColor: '#86BC25' } : {}} />
                  </div>
                  <div className="flex align-items-center justify-content-between p-3 border-1 border-gray-200 border-round-lg bg-gray-50">
                    <span className="text-sm text-gray-600"><i className="pi pi-database mr-2 text-deloitte" />Demo 資料</span>
                    <Tag severity={configStatus?.demo_data_loaded ? "info" : "warning"} value={configStatus?.demo_data_loaded ? "已載入" : "無資料"} className="text-xs" />
                  </div>
                </div>
              </div>

              <div className="mb-5 op-80">
                <h4 className="text-gray-500 mb-2 font-medium text-sm">進階選項</h4>
                <p className="text-gray-400 text-xs mb-3">上傳自定義 Excel/PDF 以覆蓋目前的 Demo 資料。</p>
                <FileUpload
                  name="file"
                  accept=".xlsx, .xls, .pdf, .txt"
                  customUpload
                  uploadHandler={(e) => { handleFileUpload(e); hide(); }}
                  auto
                  mode="basic"
                  chooseLabel="上傳自定義檔案"
                  className="w-full text-sm p-button-outlined border-deloitte text-deloitte"
                />
              </div>
            </div>

            <div className="p-3 border-top-1 border-gray-200 text-center">
              <small className="text-gray-400">稽核 AI 助手 v1.0</small>
            </div>
          </div>
        )}
      >
      </Sidebar>

      {/* Main Chat Area */}
      <div className="flex-1 overflow-hidden relative flex flex-column">
        <div className="flex-1 overflow-hidden relative">
          <ScrollPanel style={{ width: '100%', height: '100%' }} className="px-2 md:px-4">
            <div className="flex flex-column gap-4 py-5 max-w-5xl mx-auto">
              <AnimatePresence>
                {messages.map((msg, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}
                  >
                    <div className={`flex gap-3 max-w-30rem md:max-w-40rem ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                      <Avatar
                        label={msg.role === 'user' ? "我" : undefined}
                        image={msg.role === 'assistant' ? "/bot-avatar.png" : undefined}
                        shape="circle"
                        size="large"
                        className={`${msg.role === 'user' ? 'bg-deloitte text-white shadow-lg' : 'bg-white shadow-sm border-1 border-gray-100'}`}
                        style={{ width: '3rem', height: '3rem', flexShrink: 0 }}
                      />
                      <div className="flex flex-column gap-1 max-w-full">
                        <span className={`text-xs text-gray-400 font-medium ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                          {msg.role === 'user' ? '您' : '稽核 AI 助手'} • {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        <div className={`p-3 md:p-4 border-round-2xl shadow-1 line-height-3 text-lg ${msg.role === 'user' ? 'bg-deloitte text-white border-top-right-radius-0' : 'bg-white text-gray-800 border-1 border-gray-100 border-top-left-radius-0'}`}>
                          {msg.role === 'user' ? (
                            <p className="m-0 white-space-pre-wrap">{msg.content}</p>
                          ) : (
                            <div className="flex flex-column gap-3">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                className="markdown-content"
                              >
                                {msg.content}
                              </ReactMarkdown>

                              {msg.image && (
                                <div className="border-round-xl overflow-hidden mt-3 shadow-2 surface-0 border-1 border-gray-100">
                                  <div className="bg-gray-50 p-2 border-bottom-1 border-gray-100 flex align-items-center gap-2">
                                    <i className="pi pi-chart-pie text-deloitte"></i>
                                    <span className="text-sm font-semibold text-gray-700">分析圖表</span>
                                  </div>
                                  <div className="p-3 bg-white flex justify-content-center">
                                    <img
                                      src={`data:image/png;base64,${msg.image}`}
                                      alt="Generated Chart"
                                      className="max-w-full h-auto block border-round-lg cursor-pointer"
                                      style={{ maxHeight: '600px', objectFit: 'contain' }}
                                      onClick={(e) => {
                                        // Simple "lightbox" effect
                                        const w = window.open("");
                                        w.document.write(e.target.outerHTML);
                                      }}
                                      title="點擊查看原圖"
                                    />
                                  </div>
                                </div>
                              )}

                              {msg.file && (
                                <a href={msg.file.startsWith('http') ? msg.file : `${API_BASE}${msg.file}`} target="_blank" rel="noopener noreferrer" className="no-underline">
                                  <Button
                                    label="下載分析報表 (Excel)"
                                    icon="pi pi-download"
                                    className="w-full bg-deloitte border-deloitte"
                                  />
                                </a>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-content-start"
                >
                  <div className="flex gap-3 max-w-30rem">
                    <Avatar image="/bot-avatar.png" shape="circle" size="large" className="bg-white shadow-sm border-1 border-gray-100" />
                    <div className="p-3 border-round-2xl bg-white surface-card border-1 border-gray-100 border-top-left-radius-0 flex align-items-center gap-2 shadow-sm">
                      <div className="flex gap-1">
                        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 1 }} className="w-2 h-2 border-circle bg-gray-400"></motion.div>
                        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 1, delay: 0.2 }} className="w-2 h-2 border-circle bg-gray-400"></motion.div>
                        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 1, delay: 0.4 }} className="w-2 h-2 border-circle bg-gray-400"></motion.div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollPanel>
        </div>

        {/* Input Area */}
        <div className="p-4 backdrop-blur-md bg-white/70 border-top-1 border-white/20">
          <div className="max-w-5xl mx-auto relative flex flex-column gap-3">

            {/* Quick Actions Removed for Manual Demo */}

            <span className="p-input-icon-right w-full">
              <i className={`pi pi-send cursor-pointer hover:text-green-600 transition-colors ${!input.trim() ? 'opacity-50' : 'text-deloitte'}`}
                onClick={() => !isLoading && input.trim() && sendMessage()} />
              <InputText
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                className="w-full border-round-2xl p-3 pl-4 text-lg shadow-1 border-none bg-white/80 focus:bg-white focus:shadow-2 transition-all"
                placeholder="請輸入您的問題或是使用上方快速指令..."
                disabled={isLoading}
              />
            </span>
            <div className="text-center">
              <small className="text-gray-400 text-xs">AI 生成內容可能會有錯誤，請自行查證重要資訊。</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
