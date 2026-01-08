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

const API_BASE = import.meta.env.VITE_API_URL || '/api';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! Please upload your PDF or Excel file to start analyzing.' }
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
        toast.current.show({ severity: 'warn', summary: 'Missing Configuration', detail: 'Please check your backend .env file.' });
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
      toast.current.show({ severity: 'success', summary: 'Success', detail: 'File processed successfully', life: 3000 });
      setMessages(prev => [...prev, { role: 'assistant', content: `**${file.name}** processed! Ask me anything about it.` }]);
      event.options.clear();
    } catch (error) {
      console.error(error);
      toast.current.show({ severity: 'error', summary: 'Error', detail: error.response?.data?.detail || 'Upload failed', life: 5000 });
      event.options.clear();
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: userMsg });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.response,
        image: res.data.image,
        file: res.data.file
      }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error.response?.data?.detail || error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const startItems = (
    <div className="flex align-items-center gap-2">
      <i className="pi pi-chart-bar text-indigo-500 text-2xl"></i>
      <span className="font-bold text-xl text-900">AI Credit Analyst</span>
    </div>
  );

  const endItems = (
    <div className="flex align-items-center gap-2">
      {configStatus?.configured ? (
        <Tag severity="success" value="Connected" icon="pi pi-check" className="mr-2" />
      ) : (
        <Tag severity="danger" value="Disconnected" icon="pi pi-times" className="mr-2" />
      )}
      <Button icon="pi pi-cog" text rounded onClick={() => setSidebarVisible(true)} className="text-gray-600 hover:text-gray-900" />
    </div>
  );

  return (
    <div className="flex flex-column h-screen bg-gradient-to-br from-indigo-50 to-blue-50 font-sans">
      <Toast ref={toast} />

      {/* Glassmorphism Header */}
      <div className="flex justify-content-between align-items-center p-3 z-5 relative backdrop-blur-md bg-white/70 border-bottom-1 border-white/20 shadow-sm sticky top-0">
        {startItems}
        {endItems}
      </div>

      {/* Sidebar for Upload */}
      <Sidebar visible={sidebarVisible} onHide={() => setSidebarVisible(false)} position="right" className="w-full md:w-25rem bg-white/95 backdrop-blur-sm shadow-2">
        <h3 className="mb-2 text-indigo-900">Document Management</h3>
        <p className="text-gray-500 mb-5 text-sm">Upload your Excel or PDF files here to begin analysis.</p>

        <FileUpload
          name="file"
          accept=".xlsx, .xls, .pdf, .txt"
          customUpload
          uploadHandler={handleFileUpload}
          auto
          chooseLabel="Select File"
          emptyTemplate={<div className="flex align-items-center justify-content-center flex-column p-4 border-2 border-dashed border-indigo-200 border-round-xl bg-indigo-50/50">
            <i className="pi pi-cloud-upload text-6xl text-indigo-300 mb-3" />
            <p className="m-0 text-gray-500 font-medium">Drag & Drop files here</p>
          </div>}
        />
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
                        label={msg.role === 'user' ? "U" : undefined}
                        image={msg.role === 'assistant' ? "/bot-avatar.png" : undefined}
                        shape="circle"
                        size="large"
                        className={`${msg.role === 'user' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-white shadow-sm border-1 border-gray-100'}`}
                        style={{ width: '3rem', height: '3rem', flexShrink: 0 }}
                      />
                      <div className="flex flex-column gap-1 max-w-full">
                        <span className={`text-xs text-gray-400 font-medium ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                          {msg.role === 'user' ? 'You' : 'Assistant'} • {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        <div className={`p-3 md:p-4 border-round-2xl shadow-1 line-height-3 text-lg ${msg.role === 'user' ? 'bg-indigo-600 text-white border-top-right-radius-0' : 'bg-white text-gray-800 border-1 border-gray-100 border-top-left-radius-0'}`}>
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
                                    <i className="pi pi-chart-pie text-indigo-500"></i>
                                    <span className="text-sm font-semibold text-gray-700">Analysis Result</span>
                                  </div>
                                  <div className="p-3 bg-white flex justify-content-center">
                                    <img
                                      src={`data:image/png;base64,${msg.image}`}
                                      alt="Generated Chart"
                                      className="max-w-full h-auto block border-round-lg cursor-pointer"
                                      style={{ maxHeight: '600px', objectFit: 'contain' }}
                                      onClick={(e) => {
                                        // Simple "lightbox" effect: open in new tab or expand
                                        const w = window.open("");
                                        w.document.write(e.target.outerHTML);
                                      }}
                                      title="Click to view full size"
                                    />
                                  </div>
                                </div>
                              )}

                              {msg.file && (
                                <a href={msg.file} target="_blank" rel="noopener noreferrer" className="no-underline">
                                  <Button
                                    label="Download Exported Data"
                                    icon="pi pi-download"
                                    severity="success"
                                    className="w-full"
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

            {/* Quick Actions (Scenario Guide) */}
            <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
              <Button
                label="🔍 Analyze Reasonableness (Q1)"
                rounded
                outlined
                size="small"
                className="white-space-nowrap bg-white/80 border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                onClick={() => {
                  const msg = "請說明授信條件合理性分析，可以怎麼執行 ?";
                  setInput(msg);
                  // Optional: auto-send
                  // sendMessage(msg); 
                }}
              />
              <Button
                label="📊 Rate Distribution (Q2)"
                rounded
                outlined
                size="small"
                className="white-space-nowrap bg-white/80 border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                onClick={() => {
                  const msg = "我想針對現有房貸明細進行利率合理性分析，請畫出利率分布圖";
                  setInput(msg);
                }}
              />
              <Button
                label="⚠️ Find Outliers (Q3)"
                rounded
                outlined
                size="small"
                className="white-space-nowrap bg-white/80 border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                onClick={() => {
                  const msg = "我看到有一些離群值，請針對離群值做分析";
                  setInput(msg);
                }}
              />
              <Button
                label="⬇️ Export Stakeholders (Q4)"
                rounded
                outlined
                size="small"
                className="white-space-nowrap bg-white/80 border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                onClick={() => {
                  const msg = "新青安與行員利率較低是合理的，請輸出利益關係人離群值明細給我";
                  setInput(msg);
                }}
              />
            </div>

            <span className="p-input-icon-right w-full">
              <i className={`pi pi-send cursor-pointer hover:text-indigo-600 transition-colors ${!input.trim() ? 'opacity-50' : 'text-indigo-500'}`}
                onClick={() => !isLoading && input.trim() && sendMessage()} />
              <InputText
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                className="w-full border-round-2xl p-3 pl-4 text-lg shadow-1 border-none bg-white/80 focus:bg-white focus:shadow-2 transition-all"
                placeholder="Type a message or use the Quick Actions above..."
                disabled={isLoading}
              />
            </span>
            <div className="text-center">
              <small className="text-gray-400 text-xs">AI can make mistakes. Please verify important information.</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
