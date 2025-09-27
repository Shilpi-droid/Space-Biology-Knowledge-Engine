import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Calendar, TestTube, Rocket, User, Microscope } from "lucide-react";

interface SearchFiltersProps {
  selectedFilters: string[];
  onFiltersChange: (filters: string[]) => void;
}

export const SearchFilters = ({ selectedFilters, onFiltersChange }: SearchFiltersProps) => {
  const filterCategories = [
    {
      title: "Experiment Type",
      icon: TestTube,
      options: [
        { id: "human", label: "Human Studies", count: 847 },
        { id: "rodent", label: "Rodent Research", count: 1205 },
        { id: "plant", label: "Plant Biology", count: 623 },
        { id: "microbe", label: "Microbiology", count: 445 },
        { id: "omics", label: "Omics Studies", count: 289 }
      ]
    },
    {
      title: "Mission/Platform",
      icon: Rocket,
      options: [
        { id: "iss", label: "International Space Station", count: 1834 },
        { id: "shuttle", label: "Space Shuttle", count: 567 },
        { id: "spacex", label: "SpaceX Missions", count: 234 },
        { id: "artemis", label: "Artemis Program", count: 45 },
        { id: "apollo", label: "Apollo Program", count: 123 }
      ]
    },
    {
      title: "Research Focus",
      icon: Microscope,
      options: [
        { id: "radiation", label: "Radiation Effects", count: 456 },
        { id: "bone", label: "Bone Density", count: 378 },
        { id: "immune", label: "Immune Function", count: 445 },
        { id: "cardiovascular", label: "Cardiovascular", count: 234 },
        { id: "metabolism", label: "Metabolism", count: 189 },
        { id: "circadian", label: "Circadian Rhythm", count: 167 }
      ]
    },
    {
      title: "Time Period",
      icon: Calendar,
      options: [
        { id: "2024", label: "2024", count: 234 },
        { id: "2023", label: "2023", count: 456 },
        { id: "2020-2022", label: "2020-2022", count: 789 },
        { id: "2015-2019", label: "2015-2019", count: 845 },
        { id: "2010-2014", label: "2010-2014", count: 523 }
      ]
    }
  ];

  const handleFilterToggle = (filterId: string) => {
    const newFilters = selectedFilters.includes(filterId)
      ? selectedFilters.filter(f => f !== filterId)
      : [...selectedFilters, filterId];
    onFiltersChange(newFilters);
  };

  const clearAllFilters = () => {
    onFiltersChange([]);
  };

  return (
    <div className="space-y-6">
      <Card className="glass-card border-border/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <User className="w-5 h-5 text-primary" />
              Filters
            </CardTitle>
            {selectedFilters.length > 0 && (
              <Badge 
                variant="outline" 
                className="cursor-pointer hover:bg-destructive/20"
                onClick={clearAllFilters}
              >
                Clear All
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {filterCategories.map((category, index) => (
            <div key={category.title}>
              <div className="flex items-center gap-2 mb-3">
                <category.icon className="w-4 h-4 text-primary" />
                <h3 className="font-medium text-sm text-foreground">
                  {category.title}
                </h3>
              </div>
              
              <div className="space-y-2 ml-6">
                {category.options.map((option) => (
                  <div key={option.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={option.id}
                      checked={selectedFilters.includes(option.id)}
                      onCheckedChange={() => handleFilterToggle(option.id)}
                      className="border-border data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    />
                    <label
                      htmlFor={option.id}
                      className="text-sm cursor-pointer flex-1 flex items-center justify-between"
                    >
                      <span className="text-foreground hover:text-primary transition-colors">
                        {option.label}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {option.count}
                      </Badge>
                    </label>
                  </div>
                ))}
              </div>
              
              {index < filterCategories.length - 1 && <Separator className="mt-4" />}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Active Filters */}
      {selectedFilters.length > 0 && (
        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle className="text-sm">Active Filters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {selectedFilters.map((filterId) => {
                const category = filterCategories.find(cat => 
                  cat.options.some(opt => opt.id === filterId)
                );
                const option = category?.options.find(opt => opt.id === filterId);
                
                return (
                  <Badge
                    key={filterId}
                    variant="default"
                    className="bg-primary/20 text-primary border-primary/30 cursor-pointer hover:bg-primary/30"
                    onClick={() => handleFilterToggle(filterId)}
                  >
                    {option?.label} ×
                  </Badge>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};