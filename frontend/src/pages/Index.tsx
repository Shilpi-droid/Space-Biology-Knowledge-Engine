import { useState } from "react";
import { Search, Zap, BookOpen, Users, TrendingUp, MessageCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { SearchFilters } from "@/components/SearchFilters";
import { PublicationCard } from "@/components/PublicationCard";
import { DataVisualization } from "@/components/DataVisualization";
import { ChatBot } from "@/components/ChatBot";
import spaceHero from "@/assets/space-hero.jpg";

const Index = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);
  const [isChatOpen, setIsChatOpen] = useState(false);

  // Mock data for demonstration
  const featuredStats = [
    { label: "Publications", value: "2,847", icon: BookOpen, trend: "+12%" },
    { label: "Active Missions", value: "15", icon: Zap, trend: "+3" },
    { label: "Researchers", value: "1,200+", icon: Users, trend: "+8%" },
    { label: "Citations", value: "45,672", icon: TrendingUp, trend: "+18%" },
  ];

  const mockPublications = [
    {
      id: "1",
      title: "Microgravity Effects on Immune System Function During Long-Duration Spaceflight",
      authors: ["Dr. Sarah Chen", "Dr. Michael Rodriguez", "Dr. Lisa Park"],
      year: 2024,
      mission: "ISS Expedition 70",
      organism: "Human",
      summary: "Comprehensive analysis of immune cell function changes in astronauts during 6-month ISS missions, revealing significant T-cell suppression and altered cytokine production.",
      tags: ["Immunology", "ISS", "Human Studies", "Long-duration"],
      citations: 127,
      datasets: ["GeneLab-GL-12047", "OSDR-142"]
    },
    {
      id: "2", 
      title: "Rodent Research-29: Tissue Engineering in Microgravity Conditions",
      authors: ["Dr. James Wilson", "Dr. Elena Kowalski"],
      year: 2024,
      mission: "SpaceX CRS-29",
      organism: "Mouse",
      summary: "Investigation of 3D tissue scaffold formation in microgravity, demonstrating enhanced vascularization and cellular organization compared to Earth controls.",
      tags: ["Tissue Engineering", "SpaceX", "Rodent Studies", "3D Bioprinting"],
      citations: 89,
      datasets: ["GeneLab-GL-11892"]
    },
    {
      id: "3",
      title: "Plant Habitat-05: Arabidopsis Root Growth Mechanisms in Variable Gravity",
      authors: ["Dr. Maria Santos", "Dr. Robert Kim"],
      year: 2023,
      mission: "ISS Expedition 69",
      organism: "Arabidopsis",
      summary: "Analysis of root gravitropism and gene expression patterns in Arabidopsis thaliana under varying gravitational conditions using the Advanced Plant Habitat.",
      tags: ["Plant Biology", "Gravitropism", "Gene Expression", "ISS"],
      citations: 156,
      datasets: ["GeneLab-GL-11567", "OSDR-138"]
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-[60vh] flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${spaceHero})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/40 to-background"></div>
        </div>
        
        <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 glow-text">
            NASA <span className="text-primary">Bioscience</span> Explorer
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-muted-foreground max-w-3xl mx-auto">
            Discover decades of space biology research. Explore experiments, analyze data, and uncover insights 
            from humanity's journey to understand life beyond Earth.
          </p>
          
          {/* Search Bar */}
          <div className="glass-card p-6 rounded-2xl max-w-2xl mx-auto">
            <div className="flex gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
                <Input
                  placeholder="Search publications, experiments, or ask a question..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-input/80 border-border/50 h-12"
                />
              </div>
              <Button className="hero-button h-12 px-8">
                Search
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {featuredStats.map((stat, index) => (
              <Card key={index} className="glass-card border-border/50">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    {stat.label}
                  </CardTitle>
                  <stat.icon className="w-5 h-5 text-primary" />
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-foreground">{stat.value}</div>
                  <p className="text-xs text-primary mt-1">
                    {stat.trend} from last month
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Main Dashboard */}
      <section className="py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-4 gap-8">
            
            {/* Filters Sidebar */}
            <div className="lg:col-span-1">
              <SearchFilters 
                selectedFilters={selectedFilters}
                onFiltersChange={setSelectedFilters}
              />
            </div>

            {/* Main Content */}
            <div className="lg:col-span-3 space-y-8">
              
              {/* Data Visualization */}
              <Card className="glass-card border-border/50">
                <CardHeader>
                  <CardTitle className="text-primary">Research Timeline & Trends</CardTitle>
                  <CardDescription>
                    Interactive visualization of NASA bioscience publications over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <DataVisualization />
                </CardContent>
              </Card>

              {/* Publications Grid */}
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold">Latest Publications</h2>
                  <div className="flex gap-2">
                    <Badge variant="secondary" className="secondary-button">
                      {mockPublications.length} results
                    </Badge>
                  </div>
                </div>
                
                <div className="grid gap-6">
                  {mockPublications.map((publication) => (
                    <PublicationCard 
                      key={publication.id} 
                      publication={publication} 
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Chat Bot */}
      <ChatBot 
        isOpen={isChatOpen}
        onToggle={() => setIsChatOpen(!isChatOpen)}
      />

      {/* Floating Action Button */}
      <Button
        onClick={() => setIsChatOpen(!isChatOpen)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full hero-button shadow-lg z-50"
      >
        <MessageCircle className="w-6 h-6" />
      </Button>
    </div>
  );
};

export default Index;