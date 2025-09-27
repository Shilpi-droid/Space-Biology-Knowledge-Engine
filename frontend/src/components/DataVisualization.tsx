import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, BarChart3, PieChart as PieChartIcon, Network } from "lucide-react";

export const DataVisualization = () => {
  const [activeChart, setActiveChart] = useState<'timeline' | 'topics' | 'missions'>('timeline');

  // Mock data for visualizations
  const timelineData = [
    { year: '2019', publications: 156, missions: 3 },
    { year: '2020', publications: 189, missions: 4 },
    { year: '2021', publications: 234, missions: 5 },
    { year: '2022', publications: 278, missions: 6 },
    { year: '2023', publications: 312, missions: 7 },
    { year: '2024', publications: 289, missions: 8 },
  ];

  const topicsData = [
    { name: 'Immune Function', value: 445, color: 'hsl(200, 100%, 50%)' },
    { name: 'Bone Density', value: 378, color: 'hsl(25, 95%, 60%)' },
    { name: 'Radiation Effects', value: 456, color: 'hsl(270, 100%, 70%)' },
    { name: 'Plant Biology', value: 623, color: 'hsl(120, 60%, 50%)' },
    { name: 'Cardiovascular', value: 234, color: 'hsl(340, 75%, 55%)' },
    { name: 'Metabolism', value: 189, color: 'hsl(60, 90%, 55%)' },
  ];

  const missionsData = [
    { mission: 'ISS', publications: 1834, experiments: 245 },
    { mission: 'Space Shuttle', publications: 567, experiments: 123 },
    { mission: 'SpaceX', publications: 234, experiments: 89 },
    { mission: 'Artemis', publications: 45, experiments: 23 },
    { mission: 'Apollo', publications: 123, experiments: 45 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card p-3 border border-border/50">
          <p className="text-foreground font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-primary">
              {entry.dataKey}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const chartButtons = [
    { id: 'timeline', label: 'Timeline', icon: TrendingUp },
    { id: 'topics', label: 'Topics', icon: PieChartIcon },
    { id: 'missions', label: 'Missions', icon: BarChart3 },
  ];

  return (
    <div>
      {/* Chart Navigation */}
      <div className="flex gap-2 mb-6">
        {chartButtons.map((button) => (
          <Button
            key={button.id}
            variant={activeChart === button.id ? "default" : "outline"}
            size="sm"
            onClick={() => setActiveChart(button.id as any)}
            className={activeChart === button.id ? "hero-button" : ""}
          >
            <button.icon className="w-4 h-4 mr-2" />
            {button.label}
          </Button>
        ))}
      </div>

      {/* Chart Container */}
      <div className="h-80">
        {activeChart === 'timeline' && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="year" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12} 
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12} 
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="publications" 
                stroke="hsl(var(--primary))" 
                strokeWidth={3}
                dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: 'hsl(var(--primary))', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        {activeChart === 'topics' && (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={topicsData}
                cx="50%"
                cy="50%"
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {topicsData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        )}

        {activeChart === 'missions' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={missionsData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="mission" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12} 
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12} 
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="publications" 
                fill="hsl(var(--primary))"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Chart Description */}
      <div className="mt-4 p-4 bg-muted/10 rounded-lg">
        <p className="text-sm text-muted-foreground">
          {activeChart === 'timeline' && "Publication trends over the past 6 years show steady growth in NASA bioscience research output."}
          {activeChart === 'topics' && "Research focus distribution across major bioscience topics in NASA's space biology programs."}
          {activeChart === 'missions' && "Publication output by mission platform, highlighting the ISS as the primary research facility."}
        </p>
      </div>
    </div>
  );
};