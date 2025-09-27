import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { X, Send, Bot, User, Sparkles, BookOpen, Rocket } from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  citations?: string[];
}

interface ChatBotProps {
  isOpen: boolean;
  onToggle: () => void;
}

export const ChatBot = ({ isOpen, onToggle }: ChatBotProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your NASA Bioscience Research Assistant. I can help you explore space biology publications, explain complex experiments, or find specific research data. What would you like to know?',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const quickQuestions = [
    "What happens to the immune system in space?",
    "Show me recent plant experiments on ISS",
    "How does microgravity affect bone density?",
    "Find studies about radiation effects"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (messageContent?: string) => {
    const content = messageContent || inputValue.trim();
    if (!content) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const botResponse = generateBotResponse(content);
      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const generateBotResponse = (userQuery: string): Message => {
    // Mock AI responses based on query content
    let response = '';
    let citations: string[] = [];

    if (userQuery.toLowerCase().includes('immune')) {
      response = 'Space flight significantly impacts immune system function. Key findings include:\n\n• **T-cell suppression**: Reduced T-cell activation and proliferation observed in astronauts\n• **Cytokine changes**: Altered inflammatory response patterns\n• **Reactivation risk**: Increased viral reactivation (HSV, VZV) during missions\n\nRecent studies show these effects begin within days of launch and can persist for months post-flight.';
      citations = ['Chen et al., 2024 - ISS Expedition 70', 'Rodriguez et al., 2023 - Immune Function Analysis'];
    } else if (userQuery.toLowerCase().includes('plant')) {
      response = 'Recent plant experiments on the ISS have revealed fascinating adaptations:\n\n• **Advanced Plant Habitat (APH)**: Current studies with Arabidopsis and cotton\n• **Root growth**: Modified gravitropism responses in microgravity\n• **Gene expression**: Upregulation of stress response genes\n• **Water management**: Improved techniques for plant hydration in space\n\nThe Plant Habitat-05 experiment is currently investigating how plants adapt their root architecture.';
      citations = ['Santos & Kim, 2023 - Plant Habitat-05', 'NASA GeneLab GL-11567'];
    } else if (userQuery.toLowerCase().includes('bone')) {
      response = 'Bone density loss is a critical concern for long-duration spaceflight:\n\n• **Loss rate**: ~1-2% per month in weight-bearing bones\n• **Recovery**: Takes 3-4 times the mission duration to recover\n• **Countermeasures**: COLPA device and exercise protocols show promise\n• **Molecular changes**: Altered osteoblast/osteoclast activity\n\nRecent research focuses on pharmaceutical interventions and advanced exercise equipment.';
      citations = ['Wilson et al., 2024 - Bone Density Study', 'ISS Medical Research'];
    } else if (userQuery.toLowerCase().includes('radiation')) {
      response = 'Space radiation poses unique challenges for biological systems:\n\n• **Cosmic rays**: High-energy particles cause DNA damage\n• **Solar events**: Unpredictable radiation spikes during solar storms\n• **Shielding**: Current ISS provides moderate protection\n• **Biomarkers**: Chromosomal aberrations and oxidative stress indicators\n\nOngoing research investigates radioprotective compounds and improved shielding materials.';
      citations = ['Radiation Biology Lab, 2024', 'Space Radiation Research'];
    } else {
      response = 'I can help you explore NASA\'s extensive bioscience research database. Try asking about:\n\n• **Specific topics**: immune function, bone density, plant growth\n• **Mission data**: ISS experiments, shuttle studies\n• **Research methods**: omics studies, microscopy data\n• **Time periods**: recent findings vs. historical data\n\nWhat specific aspect of space biology interests you most?';
    }

    return {
      id: Date.now().toString(),
      content: response,
      sender: 'bot',
      timestamp: new Date(),
      citations
    };
  };

  if (!isOpen) return null;

  return (
    <Card className="fixed bottom-20 right-6 w-96 h-[500px] glass-card border-border/50 shadow-2xl z-40 flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-primary">
            <Sparkles className="w-5 h-5" />
            Research Assistant
          </CardTitle>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onToggle}
            className="h-8 w-8 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-4 pt-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-2 max-w-[80%] ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.sender === 'user' ? 'bg-primary' : 'bg-muted'
                }`}>
                  {message.sender === 'user' ? (
                    <User className="w-4 h-4 text-primary-foreground" />
                  ) : (
                    <Bot className="w-4 h-4 text-foreground" />
                  )}
                </div>
                
                <div className={`rounded-lg p-3 ${
                  message.sender === 'user' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'bg-muted/30 text-foreground'
                }`}>
                  <div className="text-sm whitespace-pre-line">
                    {message.content}
                  </div>
                  
                  {message.citations && (
                    <div className="mt-2 pt-2 border-t border-border/30">
                      <div className="text-xs text-muted-foreground mb-1">Sources:</div>
                      {message.citations.map((citation, index) => (
                        <Badge key={index} variant="outline" className="text-xs mr-1 mb-1">
                          <BookOpen className="w-3 h-3 mr-1" />
                          {citation}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start">
              <div className="flex gap-2">
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="w-4 h-4 text-foreground" />
                </div>
                <div className="bg-muted/30 rounded-lg p-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        {messages.length === 1 && (
          <div className="mb-4">
            <div className="text-xs text-muted-foreground mb-2">Try asking:</div>
            <div className="flex flex-wrap gap-1">
              {quickQuestions.map((question, index) => (
                <Badge
                  key={index}
                  variant="outline"
                  className="cursor-pointer hover:bg-primary/20 text-xs"
                  onClick={() => handleSendMessage(question)}
                >
                  {question}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Input */}
        <div className="flex gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about space biology research..."
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            className="flex-1 bg-input/80 border-border/50"
          />
          <Button 
            onClick={() => handleSendMessage()}
            className="hero-button"
            disabled={!inputValue.trim() || isTyping}
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};