"use client";

import { useState, useEffect } from "react";
import { Search, Filter, Download, Pill, Navigation, MessageSquare, Bell, AlertTriangle, Calendar, Clock, Heart, Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input } from "@/components/ui";
import { cn } from "@/lib/utils";
import { getActivities, type ActivityLog } from "@/lib/dashboard-api";

const activityTypes = ["All", "Medication", "Navigation", "Conversation", "Alert"];

// ---------------------------------------------------------------------------
// Fallback mock data — used ONLY when backend returns empty/error
// ---------------------------------------------------------------------------
const FALLBACK_ACTIVITIES: ActivityLog[] = [
  { id: "1", rover_id: "", type: "medication", title: "Medication Taken", description: "Took Metformin (500mg)", timestamp: "2026-01-19T12:10:00Z" },
  { id: "2", rover_id: "", type: "navigation", title: "Navigation Complete", description: "Navigated from Living Room to Kitchen", timestamp: "2026-01-19T11:30:00Z" },
  { id: "3", rover_id: "", type: "conversation", title: "Conversation", description: "Had a 15-minute chat with NovaCare about the weather", timestamp: "2026-01-19T10:45:00Z" },
  { id: "4", rover_id: "", type: "medication", title: "Medication Taken", description: "Took Lisinopril (10mg)", timestamp: "2026-01-19T08:05:00Z" },
  { id: "5", rover_id: "", type: "alert", title: "Low Battery Warning", description: "Rover battery at 15% - Now charging", timestamp: "2026-01-19T07:30:00Z" },
  { id: "6", rover_id: "", type: "navigation", title: "Navigation Complete", description: "Navigated from Bedroom to Bathroom", timestamp: "2026-01-19T07:15:00Z" },
  { id: "7", rover_id: "", type: "alert", title: "Fall Detection", description: "Possible fall detected in hallway - Patient confirmed OK", timestamp: "2026-01-18T18:45:00Z" },
  { id: "8", rover_id: "", type: "medication", title: "Medication Taken", description: "Took Aspirin (81mg)", timestamp: "2026-01-18T18:05:00Z" },
];

const typeIcons: Record<string, typeof Pill> = {
  medication: Pill,
  navigation: Navigation,
  conversation: MessageSquare,
  alert: AlertTriangle,
  vital: Heart,
};

const typeColors: Record<string, { bg: string; text: string; icon: string }> = {
  medication: { bg: "bg-success-100 dark:bg-success-900/30", text: "text-success-700 dark:text-success-400", icon: "bg-success" },
  navigation: { bg: "bg-primary-100 dark:bg-primary-900/30", text: "text-primary-700 dark:text-primary-400", icon: "bg-primary" },
  conversation: { bg: "bg-purple-100 dark:bg-purple-900/30", text: "text-purple-700 dark:text-purple-400", icon: "bg-purple-500" },
  alert: { bg: "bg-secondary-100 dark:bg-secondary-900/30", text: "text-secondary-700 dark:text-secondary-400", icon: "bg-secondary" },
  vital: { bg: "bg-accent-100 dark:bg-accent-900/30", text: "text-accent-700 dark:text-accent-400", icon: "bg-accent" },
};

function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

export default function ActivityPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("All");
  const [isLoading, setIsLoading] = useState(true);
  const [activities, setActivities] = useState<ActivityLog[]>(FALLBACK_ACTIVITIES);

  useEffect(() => {
    async function fetchActivities() {
      setIsLoading(true);
      try {
        const res = await getActivities();
        if (res.status === "success" && res.data && res.data.length > 0) {
          setActivities(res.data);
        }
        // If empty, keep fallback data
      } catch (error) {
        console.error("Failed to fetch activities:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchActivities();
  }, []);

  const filteredActivities = activities.filter((activity) => {
    const matchesSearch =
      activity.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      activity.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType =
      selectedType === "All" || activity.type.toLowerCase() === selectedType.toLowerCase();
    return matchesSearch && matchesType;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Filters */}
      <Card variant="elevated">
        <CardContent className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <Input
              placeholder="Search activities..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-5 h-5" />}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {activityTypes.map((type) => (
              <button
                key={type}
                onClick={() => setSelectedType(type)}
                className={cn(
                  "px-4 py-2 rounded-xl text-sm font-medium transition-all",
                  selectedType === type
                    ? "bg-primary text-white"
                    : "bg-gray-100 dark:bg-gray-700 text-text-secondary dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                )}
              >
                {type}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <Button className="dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<Calendar className="w-4 h-4 " />}>
              Date Range
            </Button>
            <Button className="dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<Download className="w-4 h-4" />}>
              Export
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Activity Timeline */}
      <Card variant="elevated">
        <CardHeader>
          <CardTitle>Activity Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
            </div>
          ) : (
            <div className="space-y-1">
              {filteredActivities.length > 0 ? (
                filteredActivities.map((activity, index) => {
                  const colors = typeColors[activity.type] || typeColors.alert;
                  const Icon = typeIcons[activity.type] || Bell;
                  const currentDate = formatDate(activity.timestamp);
                  const prevDate = index > 0 ? formatDate(filteredActivities[index - 1].timestamp) : null;
                  const showDateHeader = index === 0 || prevDate !== currentDate;

                  return (
                    <div key={activity.id}>
                      {showDateHeader && (
                        <div className="flex items-center gap-3 py-4">
                          <div className="h-px flex-1 bg-gray-200 dark:bg-gray-700" />
                          <span className="text-sm font-medium text-text-muted dark:text-gray-400 px-3">
                            {currentDate}
                          </span>
                          <div className="h-px flex-1 bg-gray-200 dark:bg-gray-700" />
                        </div>
                      )}
                      <div className="flex gap-4 p-4 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                        <div className="relative">
                          <div
                            className={cn(
                              "w-12 h-12 rounded-xl flex items-center justify-center",
                              colors.icon
                            )}
                          >
                            <Icon className="w-6 h-6 text-white" />
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-start justify-between">
                            <div>
                              <p className="font-semibold text-text-primary dark:text-white">{activity.title}</p>
                              <p className="text-text-secondary dark:text-gray-300 mt-1">{activity.description}</p>
                            </div>
                            <div className="text-right">
                              <div className="flex items-center gap-1 text-sm text-text-muted dark:text-gray-400">
                                <Clock className="w-4 h-4" />
                                {formatTime(activity.timestamp)}
                              </div>
                              <Badge className={cn("mt-2", colors.bg, colors.text)}>
                                {activity.type}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
                    <Search className="w-8 h-8 text-text-muted dark:text-gray-400" />
                  </div>
                  <p className="text-text-muted dark:text-gray-400">No activities found matching your filters</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
