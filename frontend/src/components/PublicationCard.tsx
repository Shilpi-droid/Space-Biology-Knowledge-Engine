import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Download, Quote, Users, Calendar, Rocket } from "lucide-react";

interface Publication {
  id: string;
  title: string;
  authors: string[];
  year: number;
  mission: string;
  organism: string;
  summary: string;
  tags: string[];
  citations: number;
  datasets: string[];
}

interface PublicationCardProps {
  publication: Publication;
}

export const PublicationCard = ({ publication }: PublicationCardProps) => {
  return (
    <Card className="glass-card border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-lg">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-foreground hover:text-primary cursor-pointer transition-colors leading-tight mb-2">
              {publication.title}
            </h3>
            
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground mb-3">
              <div className="flex items-center gap-1">
                <Users className="w-4 h-4" />
                <span>{publication.authors.slice(0, 2).join(", ")}</span>
                {publication.authors.length > 2 && (
                  <span>+{publication.authors.length - 2} more</span>
                )}
              </div>
              
              <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                <span>{publication.year}</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Rocket className="w-4 h-4" />
                <span>{publication.mission}</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Quote className="w-4 h-4" />
                <span>{publication.citations} citations</span>
              </div>
            </div>
          </div>
          
          <Badge variant="outline" className="shrink-0">
            {publication.organism}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        <CardDescription className="text-base text-foreground/80 mb-4 leading-relaxed">
          {publication.summary}
        </CardDescription>
        
        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {publication.tags.map((tag, index) => (
            <Badge
              key={index}
              variant="secondary"
              className="bg-muted/30 text-muted-foreground hover:bg-primary/20 hover:text-primary cursor-pointer transition-colors"
            >
              {tag}
            </Badge>
          ))}
        </div>
        
        {/* Datasets */}
        {publication.datasets.length > 0 && (
          <div className="bg-muted/20 rounded-lg p-3 mb-4">
            <h4 className="text-sm font-medium text-foreground mb-2">Available Datasets:</h4>
            <div className="flex flex-wrap gap-2">
              {publication.datasets.map((dataset, index) => (
                <Badge
                  key={index}
                  variant="outline"
                  className="text-xs bg-card border-primary/30 text-primary hover:bg-primary/10 cursor-pointer"
                >
                  {dataset}
                </Badge>
              ))}
            </div>
          </div>
        )}
        
        {/* Action Buttons */}
        <div className="flex gap-3">
          <Button variant="outline" size="sm" className="flex-1">
            <ExternalLink className="w-4 h-4 mr-2" />
            View Full Paper
          </Button>
          <Button variant="outline" size="sm" className="flex-1">
            <Download className="w-4 h-4 mr-2" />
            Download Data
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};